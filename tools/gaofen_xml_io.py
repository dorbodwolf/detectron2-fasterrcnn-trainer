#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import ElementTree as ET, Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'

class GaofenXMLWriter:

    def __init__(self, filename):
        self.filename = filename
        self.boxlist = []
        self.scorelist = []
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        source = SubElement(top, 'source')
        filename = SubElement(source, 'filename')
        filename.text = self.filename
        origin = SubElement(source, 'origin')
        origin.text = 'GF2/GF3'

        research = SubElement(top, 'research')
        version = SubElement(research, 'version')
        version.text = '4.0'
        provider = SubElement(research, 'provider')
        provider.text = '明溪梦之队'                
        author = SubElement(research, 'author')
        author.text = '明溪梦之队'
        pluginname = SubElement(research, 'pluginname')
        pluginname.text = '桥梁目标识别'
        pluginclass = SubElement(research, 'pluginclass')
        pluginclass.text = '识别'
        time = SubElement(research, 'time')
        time.text = '2020-07-2020-11'

        objects = SubElement(top, 'objects')

        return top

    def addBndBox(self, xmin, ymin, xmax, ymax):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        self.boxlist.append(bndbox)

    def addScore(self, score):
        self.scorelist.append(score)

    def appendObjects(self, top):
        vars = top.findall('objects')
        objects = vars[0]
        for i, each_object in enumerate(self.boxlist):
            xmin = each_object['xmin']
            xmax = each_object['xmax']
            ymin = each_object['ymin']
            ymax = each_object['ymax']

            object_item = SubElement(objects, 'object')

            coordinate = SubElement(object_item, 'coordinate')
            coordinate.text = 'pixel'
            type_ = SubElement(object_item, 'type')
            type_.text = 'rectangle'
            description = SubElement(object_item, 'description')
            description.text = 'None'

            possibleresult = SubElement(object_item, 'possibleresult')
            name = SubElement(possibleresult, 'name')
            name.text = 'bridge'
            probability = SubElement(possibleresult, 'probability')
            probability.text = str(self.scorelist[i])

            points = SubElement(object_item, 'points')
            point1 = SubElement(points, 'point')
            point1.text =  str(xmin) + ', ' + str(ymax)
            point2 = SubElement(points, 'point')
            point2.text =  str(xmax) + ', ' + str(ymax)
            point3 = SubElement(points, 'point')
            point3.text =  str(xmax) + ', ' + str(ymin)
            point4 = SubElement(points, 'point')
            point4.text =  str(xmin) + ', ' + str(ymin)
            point5 = SubElement(points, 'point')
            point5.text =  str(xmin) + ', ' + str(ymax)

    def save(self, targetFile=None):
        root = self.genXML()
        # print(root)
        self.appendObjects(root)
        # out_file = None
        # if targetFile is None:
        #     out_file = codecs.open(
        #         self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        # else:
        #     out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        # print(prettifyResult)

        root_prettify = ElementTree.fromstring(prettifyResult)
        
        # tree = etree.ElementTree(root)
        tree = ET(element=root_prettify)
        tree.write(targetFile, xml_declaration=True, encoding="utf-8")
        # out_file.write(prettifyResult.decode('utf8'),  method='xml')
        # out_file.close()