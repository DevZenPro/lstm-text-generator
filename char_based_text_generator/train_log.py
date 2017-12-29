import pickle
from datetime import datetime

from bokeh.plotting import figure, show
#from ..bokeh4github import show
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter
from bokeh.layouts import column


class Log():
    def __init__(self, cnfg):
        self.cnfg = cnfg
        self.save_as = cnfg.log_save_as
        
        self.steps = []
        self.ll = []
        self.accu = []
        self.end_time = []
        
        self.save()
        
    def save(self):
        pickle.dump(self, open(self.save_as, 'wb'))   
        
    def record(self, step, ll, accu):
        self.steps.append(step)
        self.ll.append(ll)
        self.accu.append(accu)        
        self.end_time.append(datetime.now())
        
        self.save()
        
    def get_accuracy_graph(self):
        p = figure(title='Accuracy', plot_width=1000, plot_height=400)
        p.line(self.steps, self.accu, line_width=3, color='#af9880')  # Light brown
                    
        p.xaxis.axis_label = 'steps'
        p.yaxis.axis_label = 'accuracy'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = NumeralTickFormatter(format='0.00%')
        
        return p
        
    def get_logloss_graph(self):
        p = figure(title='Logloss', plot_width=1000, plot_height=400)
        p.line(self.steps, self.ll, line_width=3, color='#FD367E')  # Dark pink

        p.xaxis.axis_label = 'steps'
        p.yaxis.axis_label = 'logloss'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = PrintfTickFormatter(format='%.3f')
        
        return p
        
    def show_progress(self, accuracy=True, logloss=True):
        if accuracy and logloss:
            accuracy_graph = self.get_accuracy_graph()
            logloss_graph = self.get_logloss_graph()

            p = column(accuracy_graph, logloss_graph)
            show(p)
            return
        
        if accuracy:
            show(self.get_accuracy_graph())
            return
        
        if logloss:
            show(self.get_logloss_graph())
            return
