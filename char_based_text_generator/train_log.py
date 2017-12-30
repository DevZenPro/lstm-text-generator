import pymongo
import numpy as np

from bokeh.plotting import figure, show
from .bokeh4github import show
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter
from bokeh.layouts import column


class Log():
    def __init__(self, log_collection_name):
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client['lastfm_x_wikipedia']
        self.coll = db[log_collection_name]

    def get_log_data(self, field1, field2, n_every):
        list1 = []
        list2 = []
        for doc in self.coll.find({'$or': [{'step': 1}, {'step': {'$mod': [n_every, 0]}}]}):
            list1.append(doc[field1])
            list2.append(doc[field2].to_decimal())    
        return list1, list2        
        
    def get_average(self, list_, n):
        ave = []
        for i in range(len(list_)):
            section = list_[max(0, i-n+1): i+1]
            ave.append(np.array(section).mean())
        return ave
    
    def get_graph(self, dct):
        p = figure(title=dct['title'], plot_width=1000, plot_height=400)
        p.line(dct['x']['data'], dct['y']['data'], line_width=3, color=dct['y']['color'])
        p.line(dct['x']['data'], dct['ave y']['data'], line_width=3, color=dct['ave y']['color'])
                    
        p.xaxis.axis_label = dct['x axis']['label']
        p.yaxis.axis_label = dct['y axis']['label']
        p.xaxis.formatter = dct['x axis']['formatter']
        p.yaxis.formatter = dct['y axis']['formatter']
        
        return p
        
    def get_accuracy_graph(self, n_every=1000, n_ave=10):  
        steps, accu = self.get_log_data('step', 'accuracy', n_every)
        ave_accu = self.get_average(accu, n_ave)
        
        dct = {'title': 'Accuracy', 
               'x': {'data': steps}, 
               'y': {'data': accu, 'color': '#efeae5'},  # Light brown
               'ave y': {'data': ave_accu, 'color': '#af9880'},  # Brown
               'x axis': {'label': 'steps', 'formatter': NumeralTickFormatter(format='0,000')},
               'y axis': {'label': 'accuracy', 'formatter': NumeralTickFormatter(format='0.00%')}}
        
        return self.get_graph(dct)
    
    def get_logloss_graph(self, n_every=1000, n_ave=10):
        steps, ll = self.get_log_data('step', 'logloss', n_every)
        ave_ll = self.get_average(ll, n_ave)
        
        dct = {'title': 'Logloss', 
               'x': {'data': steps}, 
               'y': {'data': ll, 'color': '#fed6e5'},  # Light pink
               'ave y': {'data': ave_ll, 'color': '#FD367E'},  # Pink
               'x axis': {'label': 'steps', 'formatter': NumeralTickFormatter(format='0,000')},
               'y axis': {'label': 'logloss', 'formatter': PrintfTickFormatter(format='%.3f')}}              

        return self.get_graph(dct)
        
    def show_progress(self, n_every=1000, n_ave=10, accuracy=True, logloss=True):
        self.print_most_recent_generated()
        
        if accuracy and logloss:
            accuracy_graph = self.get_accuracy_graph(n_every, n_ave)
            logloss_graph = self.get_logloss_graph(n_every, n_ave)

            p = column(accuracy_graph, logloss_graph)
            show(p)
            return
        
        if accuracy:
            show(self.get_accuracy_graph(n_every, n_ave))
            return
        
        if logloss:
            show(self.get_logloss_graph(n_every, n_ave))
            return

    def print_most_recent_generated(self):
        doc = self.coll.find({'generated': {'$exists': True}}).sort('step', pymongo.DESCENDING)[0]
        print('step {:,}:'.format(doc['step']))
        print('-' * 98)
        print(doc['generated'].replace('\n', '\n\n'))
        
        