def micro_fraction_of_new_total(f_target, f_base=0.24, ra=0.34, rv=0.18):
    k = f_target / f_base
    return 1 - (ra + rv)/k  # = (k - (ra+rv))/k

targets = [0.24, 0.43, 0.62, 0.81]
for target in targets:
    print(f'{target}: R_micro mult. = {micro_fraction_of_new_total(target)/0.48}')

