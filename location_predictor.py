import numpy as np
import pickle

def predict_location(feat):
  filename = 'finalized_model.sav'
  model=pickle.load(open(filename, 'rb'))
  train= model.predict_proba(feat)

  x=[16,230,352,474,596,702,702,596,474,352,
  230,108,2,124,246,368,490,612,734,856,
  856,135,204,326,448,631,753,875,875,753,
  753,631,570,110,232,354,476,598,781,781,
  781,781,781,781,659,537,415,293,171,49,
  171,171,171,171,293,415,415,415,415,537,
  537,537,537,659,659,659,537,135,257,257,
  257,257,257,562,562,562,562,562,745,745,
  745,745,745,255,15,493 
  ]
  y=[801,801,801,801,801,801,684,684,684,684,
  684,684,684,440,440,440,440,440,440,440,
  196,34,34,34,34,34,34,34,156,156,
  178,156,278,763,763,763,763,763,702,580,
  458,336,214,92,92,92,92,92,92,92,
  214,336,458,580,580,519,397,275,214,214,
  336,458,580,519,397,275,275,40,40,223,
  406,589,772,772,589,406,223,40,40,223,
  406,589,711,820,820,820
  ]
  z=[63,63,63,63,63,63,63,63,63,63,
  63,63,63,63,63,63,63,63,63,63,
  63,63,63,63,63,63,63,63,63,63,
  63,63,63,20,20,20,20,20,20,20,
  20,20,20,20,20,20,20,20,20,20,
  20,20,20,20,20,22,22,22,22,20,
  20,20,20,20,20,20,20,188,188,188,
  188,188,188,188,188,188,188,188,188,188,
  188,188,188,134,134,134]
  j=0
  res=0
  resy=0
  resz=0

  for num in train[0]:
    if num!= 0:
      resy=num*y[j]+resy
      res=num*x[j]+res    
      resz=num*z[j]+resz
    j=j+1
  return np.array([res, resy, resz])
