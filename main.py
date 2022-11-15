"""
Ali Murtaza Malik 

"""
# %%
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

# %% Plotting parameters
# plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %% Load data
data = np.loadtxt('./COVID19QC.csv', dtype=float,
                  delimiter=',', skiprows=2, usecols=(2,))
y = data.copy()

#%%
# Plot the raw COVID-19 data, new cases per day versus time
N = data.size
t = np.linspace(0, N, N, endpoint=True)
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5, forward=True)
ax.set_title(r'New COVID-19 Cases per Day versus Time')
ax.set_xlabel(r'$t$ (days)')
ax.set_ylabel(r'$y_k$ (cases)')
plt.plot(t, y, label='New Cases per Day')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# fig.savefig('figs/COVID19_cases_vs_time.pdf')

# %% AR model functions
def form_A_b(y, ell):
    
    #get number of elements in y
    N = y.size

    #initialize A and b matrices, we already know their size from Q3a.
    A = np.zeros((N-ell, ell))
    b = np.zeros((N-ell,1))

    #initialize the number of rows and columns as different variables
    num_rows, num_cols = A.shape

    #for my convenience
    l = ell

    #iterate over rows of A
    for r in range(num_rows):

        #to compute each row of A as shown in Q3a.
        #each subsequent row will start at the next r value and terminate at the new l value as required
        A[r] = y[r:l]
        l+=1
    
    #a counter basically
    row_b=0  

    #for my convenience, define l again instead of working with ell
    l_b=ell

    #to compute y_k (see my definition for y_k in Q3a)

    #since y_k starts at l+1 abd terimates at N
    for l_value_b in range(l_b, N):
    
        b[row_b] = y[l_value_b]

        row_b+=1

    return A, b

def fit(A, b):
    # This function solves Ax = b where x are the AR model parameters
    return linalg.solve(A.T @ A, A.T @ b)


def predict(y, beta):
    # parameters and data at each time step. 
    N = y.size
    ell = beta.size
    
    #initialize the A matrix
    A = np.zeros((N-ell, ell))

    #define rows and columns
    num_rows, num_cols = A.shape

    l = ell

    #computing A matrix again (same procedure as form_A_b function above)
    for r in range(num_rows):
        A[r] = y[r:l]
        l+=1

    #we get y_k in matrix form my matrix multiplication of our A matrix and beta matrix
    y_matrix = A @ beta 

    #convert y_k in matrix form to a 1D array for plotting purposes
    y_pred = y_matrix.flatten()
    
    # Set the first ell predictions to the average of the first ell measurments
    y_pred_mean = np.mean(y[:ell])
    y_pred = np.append((np.ones(ell) * y_pred_mean), y_pred)  
    
    return y_pred


# %% Fit (train) the AR model
N_start = 49  # start day, don't change
N_end = 399  # end day, don't change
ell = 16  # memory of AR model, change to 14, 15, 16
t = np.arange(N_start, N_end)  # time, don't change

y_scale = np.max(y[N_start:N_end])  
y = y / y_scale  # non-dimensionalize the data
A, b = form_A_b(y[N_start:N_end], ell)  # form A, b matrices
beta = fit(A, b)  # find the beta parameters

y = y * y_scale  # dimensionalize the data again
y_true = y[N_start:N_end]   #get y_true
y_pred = predict(y[N_start:N_end], beta)  #get y_pred
e = np.abs(y_true-y_pred)   #computing the error

# Plotting
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r'AR Model Train')
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (days)')
ax[0].set_ylabel(r'$y_k$ (cases)')
ax[1].set_ylabel(r'$e_k$ (cases)')
ax[0].plot(t[ell:], y_pred[ell:], label=r'$y_{k, pred, \ell=%s}$' % ell)
ax[0].plot(t[ell:], y[N_start + ell:N_end], '--', label=r'$y_{k, true}$')
ax[1].plot(t[ell:], e[ell:], label=r'$e_{k, \ell=%s}$' % ell)
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()
# fig.savefig('figs/AR_response_train.pdf')


# %% Test
y = data.copy()
N_start = 400  # start day, don't change
N_end = 625  # end day, don't change
t = np.arange(N_start, N_end)  # time, don't change

y_pred = predict(y[N_start:N_end], beta)  # predictions

# Compute various metrics associated with prediction error
y_true = y[N_start:N_end]   #get y_true
e = np.abs(y_true-y_pred)   #get absolute error
e_rel = (e/y_true)*100      #ger relative error
mu = np.mean(e)  #get mean of the absolute error
sigma = np.std(e)  #get sd of the absolute error
mu_e_rel = np.mean(e_rel) #get mean of the rel error
sigma_e_rel = np.std(e_rel)  #get sd of the absolute error


fig, ax = plt.subplots(3, 1)
# Format axes
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r'AR Model Test')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (days)')
ax[0].set_ylabel(r'$y_k$ (cases)')
ax[1].set_ylabel(r'$e_k$ (cases)')
ax[2].set_ylabel(r'$e_{k, rel}$ (%)')
ax[0].plot(t[ell:], y_pred[ell:], label=r'$y_{k, pred, \ell=%s}$' % ell)
ax[1].plot(t[ell:], e[ell:], label=r'$e_{k, \ell=%s}$' % ell)
ax[0].plot(t[ell:], y[N_start + ell:N_end], '--', label=r'$y_{k, true}$')
ax[2].plot(t[ell:], e_rel[ell:], label=r'$e_{k, rel, \ell=%s}$' % ell)
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
plt.show()
# fig.savefig('figs/AR_response_test_ell_%s.pdf' % ell)

print('Mean absolute error is ', mu, '\n')
print('Absolute error standard deviation is', sigma, '\n')
print('Mean relative error is ', mu_e_rel, '\n')
print('Relative error standard deviation is', sigma_e_rel, '\n')
