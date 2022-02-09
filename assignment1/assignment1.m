function [varargout] = ff(func_name, varargin)
    switch func_name
        case "Feature_Scaling"
            X = varargin{1};
            [X_norm, mu, sigma] = Feature_Scaling(X);
            varargout{1} = X_norm;
            varargout{2} = mu;
            varargout{3} = sigma;
        case "Cost_Function_and_Gradient"
            X = varargin{1};
            y = varargin{2};
            theta = varargin{3};
            [J,G] = Cost_Function_and_Gradient(X, y, theta);
            varargout{1} = J;
            varargout{2} = G;
        case "Gradient_Descent"
            X = varargin{1};
            y = varargin{2};
            theta = varargin{3};
            alpha = varargin{4};
            N = varargin{5};
            [theta, J_hist] = Gradient_Descent(X, y, theta, alpha, N);
            varargout{1} = theta;
            varargout{2} = J_hist;
    end
    
    function [X_norm, mu, sigma] = Feature_Scaling(X)
        % [X_norm] = Feature_Scaling(X) returns the standarized data points as
        % X_norm, the mean mu and standard deviation sigma.
        
        %%%%%%%%%%%  BEGIN SOLUTION %%%%%%%%%%%%%%%%%
        % Your code goes here
        mu = mean(X);
        sigma = std(X);
        X_norm = (X-mu) ./ sigma;
        %%%%%%%%%%%%%% END SOLUTION  %%%%%%%%%%%%%%%%
    
    end
    
    function [J,G] = Cost_Function_and_Gradient(X, y, theta)
        % [J,G] = Cost_Function_and_Gradient(X, y, theta) computes the cost
        % function and the gradient of the data points X and y, given the
        % parameters theta. The function returns the cost function J (scalar)
        % and the gradient G (vector with same dimensions as theta)
        
        %%%%%%%%%%%  BEGIN SOLUTION %%%%%%%%%%%%%%%%%
        % Your code goes here
        m = length(y);
        X = [ones(m,1), X];
        J = 1/(2*m) * (X*theta - y)' * (X*theta - y);   % Scalar
        
        % Calculate gradient
        G = 1/m * X' * (X*theta - y);
        %%%%%%%%%%%%%% END SOLUTION  %%%%%%%%%%%%%%%%
    
    end
    
    function [theta, J_hist] = Gradient_Descent(X, y, theta, alpha, N)
        % [theta, J_hist] = Gradient_Descent(X, y, theta, alpha, N) performs
        % gradient descent for a linear regression with parameters theta to fit
        % the data points in X and y, in N iterations and with a learning rate
        % of alpha. The function returns the optimised parameters theta and
        % value of the cost function J for every iteration.
    
        %%%%%%%%%%%  BEGIN SOLUTION %%%%%%%%%%%%%%%%%
        % Write a loop for gradient descent - with each iteration you should 
        % 1. update theta and 
        % 2. append J_hist with the new cost value
        % Your code goes here
        m = length(y);
        J_hist = zeros(N,1);
        X = [ones(m,1), X];

        for i = 1:N
            J = 1/(2*m) * (X*theta - y)' * (X*theta - y);   % Scalar
            G = 1/m * X' * (X*theta - y);
            theta = theta - alpha * G;
            J_hist(i) = J;
            disp(["J is: ", J]);
        end
        %%%%%%%%%%%%%% END SOLUTION  %%%%%%%%%%%%%%%%
    end
end
