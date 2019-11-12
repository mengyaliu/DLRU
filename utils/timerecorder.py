# -*- coding: utf-8 -*-

import os
import time

class TimeRecoder():
    def __init__(self):
        self.elapses_dict = dict()
        self.tags = []

    def get_cur_ms(self):
        now_time = time.time() * 1000
        return now_time

    def store_timestamp(self, tag, name='', index=''):
        if name != '':
            tag = tag + '$' + name
        if index != '':
            tag = tag + '#' + str(index)

        if tag in self.elapses_dict:
            self.elapses_dict[tag] = self.get_cur_ms() - self.elapses_dict[tag]
        else:
            self.elapses_dict[tag] = self.get_cur_ms()
            self.tags.append(tag)

    def show(self):
        for key in self.tags:
            print ('time: <%s>: %f ms'%(key, self.elapses_dict[key]))
