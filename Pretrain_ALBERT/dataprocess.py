txt=''
with open("datasets/dev.txt ", 'r', encoding='utf8') as f:
    for line in f.readlines():  # [:1000]:
        for line1 in line.split('.'):
            txt=txt+line1+'\n'
with open("datasets/dev1.txt ", 'a', encoding='utf8') as f1:
    f1.write(txt)