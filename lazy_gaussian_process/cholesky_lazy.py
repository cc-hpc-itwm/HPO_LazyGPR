import numpy as np
from scipy.linalg import solve_triangular

# Authors: Raju Ram <raju.ram@itwm.fraunhofer.de>
#          Sabine Mueller <sabine.b.mueller@itwm.fraunhofer.de>

class cholesky_lazy_gpr():

    """Cholesky decomposition for Lazy Gaussian process regression (GPR)"""

    def __init__(self, lagval):
        self.lag = lagval

        #number of time the full cholesky is being called
        self.num_fullchol_calls = 0
        #Lower triangular matrix obtained from lml function
        self.L_lml = 0
        #Lower triangular matrix obtained from fit function
        self.L_fit = 0
        self.iter = 0

        self.first_call_to_lml_function = False

    def cholesky(self, A, fulldecompose=True, L=0):

        """
        Performs a complete cholesky transformation
        Returns a lower triangular matrix L

        If  fulldecompose == 'True'
        We compute the full cholesky factorization with time complexity of O(n^3), i.e
        \K_{n+1} = \mathbf{L}_{n+1}\mathbf{L}_{n+1}^T

        If  fulldecompose == 'False' : We do not compute the full cholesky factorization,
        We only compute the column vector q and scalar d with a total time complexity of O(n^2)
        \mathbf{L}_{n+1} =
        \begin{pmatrix}
        \mathbf{L}_n & \mathbf{0} \\
        \mathbf{q}^T & d
        \end{pmatrix} \qquad
        """
        n = A.shape[0]

        if(fulldecompose):
            for i in range(n):
                for j in range(i):
                        A[i,j] = (A[i,j] - sum(A[i,:j]*A[j,:j]))/A[j,j]
                A[i,i] = np.sqrt(A[i,i] - sum(A[i,:i]**2))

            for i in range(n):
                for j in range(i+1,n):
                    A[i,j] = 0

            return np.matrix(A)

        else:
            nL = L.shape[0]
            num_extra_rows = n - nL

            if(num_extra_rows <= 0):
                raise ValueError('numer of extra rows should greater than 0')

            # In this for loop size of L is changing
            for additional_row in range(num_extra_rows):

                nL =  L.shape[0]

                col_A = A[0:(nL), nL]

                #compute the last row in L using the prevL
                c = np.zeros(nL + 1)

                #print(len(lastcol_A), L.shape[0])
                if( len(col_A) != nL):
                    raise ValueError('Size of L and length of colA does not match')
                c[:nL] = solve_triangular(L, col_A, lower = True)

                temp = A[nL,nL] - np.inner(c,c)

                if(temp > 0):
                    d = np.sqrt(temp)
                else:
                    # print('temp: ', temp)
                    raise ValueError('temp is expected to be greater than 0')

                c[nL] = d

                # column append
                L = np.hstack([L, np.zeros((L.shape[0],1)) ])

                #row append
                L = np.vstack([L, c])
            return L
