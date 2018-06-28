import json
import numpy as np


class ABCExecutionReport(object):

    def __init__(self, delta, n, iters, model, theta_prior, y, dim,
                 mh_sampler_iter):
        self.delta = delta
        self.n = n
        self.iters = iters
        self.model = model
        self.theta_prior = theta_prior
        self.y = y
        self.dim = dim
        self.mh_sampler_iter = mh_sampler_iter
        self.posteriors = None


class ABCExecutionReportJSON(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, ABCExecutionReport):
            json_dict = dict()
            json_dict['delta'] = o.delta
            json_dict['n'] = o.n
            json_dict['mh_sampler_iter'] = o.mh_sampler_iter
            json_dict['prior'] = (o.theta_prior.tolist()
                                  if isinstance(o.theta_prior, np.ndarray)
                                  else o.theta_prior)
            json_dict['y'] = (o.y.tolist()
                              if isinstance(o.y, np.ndarray)
                              else o.y)
            if o.posteriors is not None:
                json_dict['posteriors'] = [el.tolist()
                                           if isinstance(el, np.ndarray)
                                           else el
                                           for el in o.posteriors]
            return json_dict
        else:
            return super().default(o)
