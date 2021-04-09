import os

import xmltodict


def get_topic_list(filename):
    f = open(filename, "r")
    xml_data = f.read()
    # converting xml to dictionary
    my_dict = xmltodict.parse(xml_data)
    # determining the type of converted object
    topic_list_full = my_dict['topics']['topic']
    topic_list = [element['title'] for element in topic_list_full]
    return topic_list


def main():
    filename = "/home/yeagerists/seupd2021-yeager/topics-task-1.xml"
    topic_list = get_topic_list(filename)
    for x in topic_list:
        print(x)


main()
