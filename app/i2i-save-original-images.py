import ismrmrd
import os
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import mrdhelper
import constants
from time import perf_counter


def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    imgGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                raise Exception("Raw k-space data is not supported by this module")

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata)
                    connection.send_image(image)
                    imgGroup = []

                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

            elif item is None:
                break

            else:
                raise Exception("Unsupported data type %s", type(item).__name__)

        # Process any remaining groups of image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
        
        # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
        connection.shutdown_close()

    finally:
        try:
            connection.send_close()
        except:
            logging.error("Failed to send close message!")


def compute_nifti_affine(image_header, voxel_size):

    # Extract necessary fields
    position      = image_header.position
    read_dir      = image_header.read_dir
    phase_dir     = image_header.phase_dir
    slice_dir     = image_header.slice_dir

    # Convert from LPS to RAS
    position_ras  = [ -position[0],  -position[1],  position[2]]
    read_dir_ras  = [ -read_dir[0],  -read_dir[1],  read_dir[2]]
    phase_dir_ras = [-phase_dir[0], -phase_dir[1], phase_dir[2]]
    slice_dir_ras = [-slice_dir[0], -slice_dir[1], slice_dir[2]]

    # Construct rotation-scaling matrix
    rotation_scaling_matrix = np.column_stack([
        voxel_size[0] * np.array( read_dir_ras),
        voxel_size[1] * np.array(phase_dir_ras),
        voxel_size[2] * np.array(slice_dir_ras)
    ])

    # Construct affine matrix
    affine = np.eye(4)
    affine[:3, :3] = rotation_scaling_matrix
    affine[:3,  3] = position_ras

    return affine


def get3Darray(data) -> np.array:

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

    data_5d = data.astype(np.float64)

    # Reformat data from [y x z cha img] to [y x img]
    data_3d = data_5d[:,:,0,0,:]
    
    return data_3d


def createMRDImage(data_in, head, meta, metadata, info):

    # Reformat data from [y x img] to [y x z cha img]
    data = data_in[:,:,np.newaxis,np.newaxis,:].astype(np.int16)

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):

        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = imagesOut[iImg].data_type

        # Set the image_type to match the data_type for complex data
        if (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXFLOAT) or (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXDOUBLE):
            oldHeader.image_type = ismrmrd.IMTYPE_COMPLEX

        oldHeader.image_series_index += info['image_series_index_offset']

        imagesOut[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        if len(info['ImageProcessingHistory']) > 0:
            tmpMeta['ImageProcessingHistory']         = info['ImageProcessingHistory']
        if len(info['SequenceDescriptionAdditional']) > 0:
            tmpMeta['SequenceDescriptionAdditional']  = info['SequenceDescriptionAdditional']
        tmpMeta['Keep_image_geometry']            = 1

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    return imagesOut


def process_image(images, connection, config, metadata):
    
    if len(images) == 0:
        return []

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # OR paramter : Save Original Images
    saveoriginalimages = True
    if ('parameters' in config) and ('SaveOriginalImages' in config['parameters']):
        logging.debug(f"type of config['parameters']['SaveOriginalImages'] is {type(config['parameters']['SaveOriginalImages'])}")

        if type(config['parameters']['SaveOriginalImages']) is str:
            if   config['parameters']['SaveOriginalImages'].lower() == 'true' :
                saveoriginalimages = True
            elif config['parameters']['SaveOriginalImages'].lower() == 'false':
                saveoriginalimages = False

        elif type(config['parameters']['SaveOriginalImages']) is bool:
            saveoriginalimages = config['parameters']['SaveOriginalImages']
        
    else:
        logging.warning("config['parameters']['SaveOriginalImages'] NOT FOUND !!")
    logging.info(f'saveoriginalimages = {saveoriginalimages}')

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    logging.info(f'MRD supposed organization : [img cha z y x]')
    logging.info(f'MRD data shape : {data.shape}')
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    logging.warning(f'MRD SequenceDescription : {meta[0]['SequenceDescription']}')

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    # Diagnostic info
    matrix    = np.array(head[0].matrix_size  [:])
    fov       = np.array(head[0].field_of_view[:])
    voxelsize = fov/matrix
    read_dir  = np.array(images[0].read_dir )
    phase_dir = np.array(images[0].phase_dir)
    slice_dir = np.array(images[0].slice_dir)
    logging.info(f'MRD computed maxtrix [x y z] : {matrix   }')
    logging.info(f'MRD computed fov     [x y z] : {fov      }')
    logging.info(f'MRD computed voxel   [x y z] : {voxelsize}')
    logging.info(f'MRD read_dir         [x y z] : {read_dir }')
    logging.info(f'MRD phase_dir        [x y z] : {phase_dir}')
    logging.info(f'MRD slice_dir        [x y z] : {slice_dir}')

    # get 3D cube from MRD input data
    data_ORIG = get3Darray(data)
    logging.info(f'input data shape : {data_ORIG.shape}')

    # initialize
    images_out = []
    info = {
        'image_series_index_offset': 0,
        'ImageProcessingHistory': [],
        'SequenceDescriptionAdditional': '',
    }

    # save orig images ?
    if saveoriginalimages:
        images_ORIG = createMRDImage(data_ORIG, head, meta, metadata, info)
        images_out += images_ORIG
        info['image_series_index_offset'] += 1

    # invert contrast
    data_inv = np.abs(data_ORIG.max()-data_ORIG)
    info['ImageProcessingHistory'].append('InvertContrast')
    info['SequenceDescriptionAdditional'] += 'InvContrast'
    images_inv = createMRDImage(data_inv, head, meta, metadata, info)
    images_out += images_inv
    
    return images_out
