def logging(out: str):
    print(out)
    with open('../logs', 'a+') as o:
        o.write(out + '\n')

