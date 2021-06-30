class SVR_PerLs:
    
    """
    SVR-LS based on Percentage error 
    Fits models where the values of the target variable are positive.
    
        -- Parameter --
            C: Parameter that determines the weight of the penalization term in the model 
               (Default = 0.1)
                        
            kernel: name of the kernel that the model will use. Written in a string format.
                    (Default = "linear"). 
        
                    acceptable parameters: 
                        "linear", "poly", "polynomial", "rbf", 
                        "laplacian", "cosine".
        
                    for more information about individual kernels, visit the 
                    sklearn pairwise metrics affinities and kernels user guide.
                    
                    https://scikit-learn.org/stable/modules/metrics.html
            
            Specific kernel parameters: 

        --Methods--
            fit(X, y): Learn from the data. Returns self.

            predict(X_test): Predicts new points. Returns X_test labels.

            coef_(): Returns w and b for linear models. Otherwise, returns alpha
                (dual vector), b (intercept) and X from the dataset.

            For more information about each method, visit specific documentations.
            
        --Example-- 
            ## Load the library
            >>> from SVRLSPercent_Library import SVR_PerLs
            ...
            ## Initialize the SVR object with custom parameters
            >>> model = SVR_PerLs(C = 10, kernel = "rbf", gamma = 0.1)
            ...
            ## Use the model to fit the data
            >>> fitted_model = model.fit(X, y)
            ...
            ## Predict with the given model
            >>> y_prediction = fitted_model.predict(X_test)
            ...
            ## e.g
            >>> print(y_prediction)
            np.array([12.8, 31.6, 16.2, 90.5, 28, 1, 49.7])
    
    """
    
    def __init__(self, C = 0.1, kernel = "linear", **kernel_param):
        import numpy as np
        from numpy.linalg import inv
        from sklearn.metrics.pairwise import pairwise_kernels
        from sklearn.utils import check_X_y, check_array
        self.np = np
        self.C = C
        self.kernel = kernel
        self.pairwise_kernels = pairwise_kernels
        self.kernel_param = kernel_param
        self.check_X_y = check_X_y
        self.check_array = check_array
        self.inv = inv
        
    def fit(self, X, y):
        
        """ 
        Computes coefficients for new data prediction.
        
            --Parameters--
                X: nxm matrix that contains all data points
                   components. n is the number of points and
                   m is the number of features of each point.
                   
                y: nx1 matrix that contains labels for all
                   the points.
            
            --Returns--
                self, containing all the parameters needed to 
                compute new data points.
        """
        
        X, y = self.check_X_y(X, y)
        # hyperparameters
        np = self.np
        C = self.C 
        
        kernel = self.kernel
        pairwise_kernels = self.pairwise_kernels
        inv = self.inv
        
        # omega + upsilon
        omega_ = pairwise_kernels(X, X, kernel, **self.kernel_param) + np.identity(y.size)*((y**2)/C)
        # ones vector
        onev = np.ones(y.shape).reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        # solve for parameters
        A = np.linalg.pinv(np.block([[0, onev.T],[onev, omega_]]))
        B = np.concatenate((np.array([0]),y.reshape(-1)))
        sol =  A @ B
        
        b = sol[0]
        alpha = sol[1:]
        
        self.X = X
        self.alpha = alpha; self.b = b
        return self
        
    def predict(self, X_):
        
        """ 
        Computes coefficients for new data prediction.
        
            --Parameters--
                X: nxm matrix that contains all data points
                   components. n is the number of points and
                   m is the number of features of each point.
                   
                y: nx1 matrix that contains labels for all
                   the points.
            
            --Returns--
                self, containing all the parameters needed to 
                compute new data points.
        """
        
        pairwise_kernels = self.pairwise_kernels
        kernel_param = self.kernel_param
        kernel = self.kernel
        alpha = self.alpha
        b = self.b
        X = self.X
        
        X_ = self.check_array(X_)
        predict = alpha @ pairwise_kernels(X, X_, metric = kernel, **kernel_param) + b
        return predict
    
    
        # coefficient
        """--Returns--
                Linear:
                - weights
                - intercept
                
                Non-Linear:
                - dual vector
                - primal vectors
                - intercept
        """
    def coef_(self):
        if self.kernel == "linear":
            alpha = self.alpha; X = self.X
            w = alpha @ X
            return w, self.b
        else: 
            return self.alpha,  self.b, self.X