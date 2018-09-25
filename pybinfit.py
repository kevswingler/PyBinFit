
##############################################################
# PyBinFit
# Some fitness functions programmed in Python
# Currently includes:
# OneMax
# K-Ones
# Bivariate quadratic
# NK Model
# Knapsack packing
# K Bit Trap
# Ising network
# MaxSat
# Binary Linear Programming
#
# All are codes to take an input as a binary array and return a fitness score
# They are designed to be used as black box fitness functions for testing metaheuristic algorithms
#
# Author: Kevin Swingler kms@cs.stir.ac.uk
# Free to re-distribute with changes but please acknowledge the original author
##############################################################


import numpy as np
from functools import reduce
import itertools

##############################################################
# General processing functions
##############################################################

def scale_01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def gen_ins_binary(numsamps,cols):
    '''Generate input array of numsamps rows and cols columns of uniform random numbers from {0,1}.'''
    x=np.random.randint(0,2,(numsamps,cols))  # numsamps rows, bits cols, all 0,1 at random
    return x

def gen_all_ins_binary(cols):
    '''Generate an array containing every combination of zeros and ones over cols bits'''
    return np.array(list(map(list, itertools.product([0, 1], repeat=cols))))

def add_noise(x,var):
    '''Add normally distributed noise with a zero mean and given variance (var) to array x'''
    x=x+np.random.normal(0, var, x.size)
    return x

def zero_to_minus_one(x):
    '''Return a copy of an array in {0,1} with the 0s replaced with -1'''
    xm=x.copy()
    xm[x==0]=-1
    return xm

def add_jags(x):
    '''Take a univariate series and make it jaggy so a hill climb won't work'''
    z=x%2==0                # Boolean index of even numbers in x
    x[x%2==1]=x[x%2==1]-1   # Change values of odd values
    x[z]=x[z]+1             # Change values of original even numbers
    return x

def bin_array_to_int(x):
    '''Convert array of binary ints to single integer decimal.
    
    To use, from functools import reduce'''
        return reduce(lambda a,b: 2*a+b, x)
    

def gen_xy_array(func_class, numsamps):
    """
    Generate an input,output sample set using the given fitness function class.
    
    Generate two arrays: x,y. x is a list, numsamps in length, of randomly generated binary patterns
    y is the output, which is the result of evaluating each entry in x using the class provided
    The class func_class must be already set up with the appropriate parameters, including the number of input bits
    That means there is no parameter for this function that determines the number of input bits
    """
    x=gen_ins_binary(numsamps,func_class.cols)
    y=eval_rows(func_class,x)
    return x,y

def eval_all_in_patterns(func_class):
    '''Return every possible input pattern and its associated output using the given fitness function.
    
    Note that the instance of the fitness function must already be fully defined (including the number of inputs).
    Returns 2^p rows, where p is the number of inputs, so be careful with p>20
    '''
    x=gen_all_ins_binary(func_class.cols)
    y=eval_rows(func_class,x)
    return x,y

def eval_rows(func_class,x):
    '''Evaluate the given array of inputs using the given fitness function and return the results.'''
    return np.array(list(map(func_class.feval,x)))
    



##############################################################
# All the fitness function classes below must have:
# A member variable, cols that defines the number of inputs
# A function named feval that takes a numpy 1 x cols array as input and produces a real valued output
# All other parameters of the function must be stored in the class, not passed to eval
##############################################################



##############################################################
# Full set counting problems
##############################################################


class onemax:
    """
    One Max Fitness function.
    
    Return the number of 1s in the input pattern.
    Maximised when all inputs set to 1.
    """
    def __init__(self,cols):
        self.cols=cols

    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        return np.sum(x)


class k_ones:
    """
    K ones fitness function.
    
    Maximised when the input pattern has any k bits set to 1.
    All other patterns score less than that by the difference between the number of bits set and k
    """
    def __init__(self,cols,k):    # k is the number of 1s in the input pattern that maximises the output
        self.cols=cols
        self.k=k
        
    # Exactly k set to one
    # Evaluate a single input pattern
    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        c=np.sum(x)
        return self.cols-(abs(c-self.k))
    

##############################################################
# Quadratic problem from Pelikan and Mühlenbein BMDA paper
##############################################################

class bivar:
    '''Quadratic fitness function requiring bivariate interactions to be modelled.
    
    From M. Pelikan and H. Mühlenbein. The bivariate marginal distribution algorithm. In R. Roy,
T. Furuhashi, and P. K. Chawdhry, editors, Advances in Soft Computing - Engineering Design
and Manufacturing, pages 521–535, London, 1999. Springer-Verlag
'''
    def __init__(self,cols):
        asset cols%2==0, "There must be an even number of inputs as they are paired"
        self.cols=cols
    
    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        sum=0
        for i in range(0,self.cols/2):
            u=x[i]
            v=x[(2*i)-1]
            sum=sum+0.9-0.9*(u+v)+1.9*u*v
        return sum

##############################################################
# NK Model
##############################################################

class NK_model:
    """
    N K Landscape fitness function.
    
    In this implementation, each of the N input bits (a locus) has an associated function
    The function is specific to the bit, so there are N functions defined
    Each function is of K+1 bits - the locus bit itself and K others
    The K others are neighbours to the right, wrapping around at the end
    Each function is a random mapping from each of the 2^(K+1) combinations of the values on its inputs to R
    These are stored in an array
    The full evaluation of an input pattern of N bits is the average of the locus function outputs
    """
    def __init__(self,N,K):
        self.K=K
        self.cols=N
        self.gen_all_functions()
    
    
    
    def gen_all_functions(self):
        '''
        Generate N random function mappings of K+1 variables, so each mapping has 2^(K+1) entries.
        
        To evaluate an input of K+1 bits, convert it to a decimal and use that as the index using bin_array_to_int(x)
        feval takes that index and the index of the key input in the block to be evaluated
        Function mappings are all random.
        '''
        self.functions=np.random.rand(self.cols,2**(self.K+1))    # Array of N functions (N in NK = cols in this class) each function has 2^(K+1) values
    
    
    def NKModel_eval_block(self,a,i):
        '''
        Evaluate the function for the ith input variable, given the input value a.
        '''
        return self.functions[i,bin_array_to_int(a)]    # The function for the ith bit and its neighbours, given input a
    
    
    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        sum=0
        K=self.K
        for i in range(self.cols):
            a=x[i:i+K+1]    # The current bit plus K more to the right
            if(a.size<K+1):
                a=np.append(a,x[0:K-a.size+1])   # Append from the start to wrap if needed
            sum=sum+self.NKModel_eval_block(a,i)
        return sum/(i+1)


##############################################################
# Knapsack Packing
##############################################################

class knapsack:
    """
    Knapsack Packing fitness function.
    
    Input array is as long as list of items to try and pack
    Each entry represents an item
    1 means pack the item, 0 means don't
    Weights and values can be set at random using rand_items(), in which case they are both from [0,1) so the average weight is 5
    Set your own weights and values with set_weights_values()
    """
    def __init__(self,num_items,maxw):
        self.cols=num_items
        self.maxw=maxw
        self.rand_items()
       
    def set_weights_values(self,weights,values):
        '''Set the weights and values of the knapsack items.'''
        self.weights=weights
        self.values=values
    
    def rand_items(self):
        '''Assign each candidate item for packing a value and a weight, chosen uniformly at random from [0,1)'''
        self.values=np.random.random(self.cols)
        self.weights=np.random.random(self.cols)
    
    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        val=(self.values*x).sum()
        weight=(self.weights*x).sum()
        return val if weight<self.maxw else 0

        
##############################################################
# K Bit Traps
##############################################################

class k_bit_trap:
    """
    K Bit Trap fitness function.
    
    Concatenated set of trap functions of size k
    Each trap leads a hill climb towards all bits set to zero unless all but 1 are set
    Maximum is at all bits=1
    """
    def __init__(self,k,rep):
        self.cols=k*rep
        self.k=k

    def eval_single_ktrap(x):
        s=x.sum()
        if(self.k==s):
            c=self.k
        else:
            c=self.k-s-1;
        #print(x,k,s,c)
        return c

    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        sum=0
        for i in range(0,self.cols,self.k):
            sum+=eval_single_ktrap(x[i:i+self.k])
        return sum


##############################################################
# Ising
##############################################################
    
class ising:
    """
    Ising model fitness funcrion of any dimension.
    
    Weights are set at random from range -1 to 1
    """
    def __init__(self,nodes,dims):
        self.cols=nodes
        self.dims=dims
        self.rand_weights()
    
    # Each node has a number of connections equal to the dimensionality of the Ising model
    # They go in one direction to the next neighbour (e.g. right, down)
    def rand_weights(self):
        self.weights=np.random.random(self.cols*self.dims)*2-1
    
    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        xm=zero_to_minus_one(x)
        sum=0
        k=0
        for i in range(0,self.cols):   # For each node
            for j in range(0,self.dims):   # For each direction of connection
                if(i+j+1<self.cols):
                    to=i+j+1
                else:
                    to=i+j+1-self.cols    # Wrap around
                sum=sum+xm[i]*xm[to]*self.weights[k]
                #print(i,"to",to,"with",self.weights[k])
                k=k+1
        return sum



##############################################################
# MAXSAT
##############################################################

class SAT:
    """
    Max satisfiability fitness function.
    
    Define the problem with a list of clauses in conjunctive normal form using:
    i means clause i is true
    -1 means clause i is false
    E.g:
    1,2,3
    1,2,-3
    Means (a or b or c) AND (a or b or not c)
    You can define your own or load ones in DIMACS cnf format
    There are examples at https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html
    """
    def __init__(self,file_path):
        self.load_cnf(file_path)
        
    # Load clauses from cnf file into structure
    def load_cnf(self,filename):
        '''Load a SAT problem specified in DIMACS cnf format.
        
        See http://www.satcompetition.org/2009/format-benchmarks2009.html
        '''
        f = open(filename, 'r')
        line="c"
        while(line[0]=='c'):  # Skip comments
            line=f.readline()
        problem=line.split()         # Now we have the problem description
        self.cols=int(problem[2])          # Number of variables
        self.num_clauses=int(problem[3])       # Number of clauses
        
        line=f.readline()
        cl=line.split()
       
        cl=np.delete(cl,-1).astype(int)  # Remove final 0 from line and cast to ints
        self.clauses=np.array([cl])       # First item in array
        for line in f:
            cl=line.split()
            if(cl[0]=='%'):
                break
            cl=np.delete(cl,-1).astype(int)  # Remove final 0 from line and cast to ints   
            self.clauses=np.append(self.clauses,[cl],axis=0)
        
    def feval(self,x):
        assert x.size==self.cols, "Input x has length {0} but length {1} expected".format(x.size,self.cols)
        xm=zero_to_minus_one(x)
        rval = 0
        cl,tm=self.clauses.shape
        for curClause in range(0,cl):
            match=0
            for curTerm in range(0,tm):
                if(np.sign(xm[abs(self.clauses[curClause][curTerm])-1]) == np.sign(self.clauses[curClause][curTerm])):
                    match=1
                    break          # Once one matches, exit
            rval=rval+match    # Add 1 or 0
        return rval

    

##############################################################
# Binary Linear Program
##############################################################

class bin_lin_prog:
    """
    Binary Linear Program fitness function.
    
    Defines a binary linear program (BLP) in the form
    
    Maximise cx subject to
    Ax <= b
    x in {0,1}^p
    
    Where cx is the objective (c and x both vectors)
    Ax <= b are the constraints (A is a matrix, b is a vector)
    p is the number of elements in x (and in c and in each row of A)    
    """
    def __init__(self,objective,constraints):
        '''
        Parameters:
        objective = 1D numpy array of coefficients for objective
        constraints = 2D numpy array of coefficients for constraints.
        Each constraint row contains coefficients, a constraint value and +1 or -1 for greater or less than
        E.g. 3,-2,5,20,-1 means 3x1-2x2+5x3 < 20, or -1,0,3,10,1 means -x1+3x3 > 10
        '''
        self.objective=objective
        self.constraints=constraints
        self.cols=len(objective)
    
    def feval(self,x):
        good=True
        for c in self.constraints:
            objeval=np.dot(x,c[:-2])    # Dot product of x and coefficients in constraint
            if(c[-1]==-1 and objeval > c[-2]) or (c[-1]==1 and objeval < c[-2]):  # Look for failed constraint
                good=False
                break
        if good:
            return np.dot(x,self.objective)
        else:
            return 0
        
