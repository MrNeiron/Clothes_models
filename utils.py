def check_tuple(tup, default = (100,100)):
    if False not in [str(el).isdigit() for el in tup]:
        return tup
    if tup[0] == '(' and tup.index(',') != -1 and tup[-1] == ')':
        try:
            indexComma = tup.index(',')
            tup = (int(''.join(tup[1:indexComma])),int(''.join(tup[indexComma+1:-1])))
        except ValueError:
            print(f"Wrong value! Resized in {default}")
            return default
    else: 
        print(f"Wrong value! Resized in {default}")
        return default
    return tup