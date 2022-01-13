import os
import numpy as np


no_grad_acc_losses, *other_res = sorted(os.listdir('results_grad_acc'))

no_grad_acc_losses = np.genfromtxt('results_grad_acc/' + no_grad_acc_losses, delimiter='\n')
for r in other_res:
    current_data = np.genfromtxt('results_grad_acc/' + r, delimiter='\n')
    assert np.allclose(no_grad_acc_losses, current_data, )
    print(f'{r} OK')
print('All tests OK')
