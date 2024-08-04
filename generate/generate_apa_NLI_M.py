import os

from data_utils_apa import *

data_dir='../data/apa/'
aspect2idx = {
    "Sustainability": 0,
    "Expertise": 1,
    "Innovation": 2,
    "Marketing & Sales": 3,
    "Personnel": 4,
    "Politics & Law": 5,
    "Products & Services": 6,
    "Production & Infrastructure": 7,
    "Scandals & Economic Crime": 8,
    "Location Identification": 9,
    "Strategy & Management": 10,
    "External Function": 11,
    "Economic Performance": 12
}

(train, train_aspect_idx), (dev, dev_aspect_idx), (test, test_aspect_idx) = load_task(data_dir, aspect2idx)

print("len(train) = ", len(train))
print("len(dev) = ", len(dev))
print("len(test) = ", len(test))

train.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
dev.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
test.sort(key=lambda x:x[2]+str(x[0])+x[3][0])

data_parts = {
    'train': train,
    'dev': dev,
    'test': test
}

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def create_nli_m_datasets(data='train'):
    with open(dir_path+f"{data}_NLI_M.tsv","w",encoding="utf-8") as f:
        f.write("id\tsentence1\tsentence2\tlabel\n")
        for v in data_parts[data]:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='entity1':f.write('entity - 1')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'entity1':
                    f.write('entity - 1')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            if v[2]=='ENTITY1':f.write('entity - 1 - ')
            if len(v[3])==1:
                f.write(v[3][0]+"\t")
            else:
                f.write("transit location\t")
            f.write(v[4]+"\n")

create_nli_m_datasets('train')
create_nli_m_datasets('dev')
create_nli_m_datasets('test')
