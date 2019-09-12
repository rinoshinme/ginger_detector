"""
PascalVOC Annotation IO functions
"""
import xml.etree.ElementTree as ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


class PascalVOC(object):
    def __init__(self, foldername='', database_src=''):
        # these are only need in gen_xml
        self.foldername = foldername
        self.database_src = database_src

    @staticmethod
    def prettify(elem):
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())

    def gen_xml(self, filename, imgsize, boxes, xml_file):
        top = Element('annotation')

        # add image info
        folder_ele = SubElement(top, 'folder')
        folder_ele.text = self.foldername
        filename_ele = SubElement(top, 'filename')
        filename_ele.text = filename

        source_ele = SubElement(top, 'source')
        database_ele = SubElement(source_ele, 'database')
        database_ele.text = self.database_src

        size_ele = SubElement(top, 'size')
        width_ele = SubElement(size_ele, 'width')
        height_ele = SubElement(size_ele, 'height')
        depth_ele = SubElement(size_ele, 'depth')
        width_ele.text = str(imgsize[0])
        height_ele.text = str(imgsize[1])
        depth_ele.text = str(imgsize[2])

        segmented_ele = SubElement(top, 'segmented')
        segmented_ele.text = '0'

        # add boxes
        for box in boxes:
            object_ele = SubElement(top, 'object')
            name_ele = SubElement(object_ele, 'name')
            name_ele.text = box['name']
            pose_ele = SubElement(object_ele, 'pose')
            pose_ele.text = 'Unspecified'
            truncated_ele = SubElement(object_ele, 'truncated')
            truncated_ele.text = '0'
            difficult_ele = SubElement(object_ele, 'difficult')
            difficult_ele.text = '0'

            bndbox_ele = SubElement(object_ele, 'bndbox')
            xmin_ele = SubElement(bndbox_ele, 'xmin')
            xmin_ele.text = str(box['bndbox'][0])
            ymin_ele = SubElement(bndbox_ele, 'ymin')
            ymin_ele.text = str(box['bndbox'][1])
            xmax_ele = SubElement(bndbox_ele, 'xmax')
            xmax_ele.text = str(box['bndbox'][2])
            ymax_ele = SubElement(bndbox_ele, 'ymax')
            ymax_ele.text = str(box['bndbox'][3])

        # save to xml file
        pretty_result = self.prettify(top)

        with open(xml_file, 'wb') as outfile:
            outfile.write(pretty_result)

    @staticmethod
    def read_xml(xml_file):
        root = ElementTree.parse(xml_file)

        filename = root.find('filename').text

        # read image info
        size_obj = root.find('size')
        width = int(size_obj.find('width').text)
        height = int(size_obj.find('height').text)
        depth = int(size_obj.find('depth').text)
        imgsize = (width, height, depth)

        # read objects
        objs = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            pose = obj.find('pose').text
            truncated = obj.find('truncated').text
            difficult = obj.find('difficult').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            objs.append({
                'name': name,
                'pose': pose,
                'truncated': truncated,
                'difficult': difficult,
                'bndbox': (xmin, ymin, xmax, ymax)
            })
        return filename, imgsize, objs
