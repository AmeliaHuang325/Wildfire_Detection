import json
nb=json.load(open('02_Model training.ipynb',encoding='utf-8'))
for i,cell in enumerate(nb['cells']):
    print('CELL',i,cell['cell_type'])
    if cell['cell_type']=='code':
        print('.join(cell['source']))
    else:
        print('.join(cell['source']).strip())
    print('-'*60)

