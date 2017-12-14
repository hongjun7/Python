from numpy import arange, array, ones
from pylab import plot, show
from scipy import stats

x = arange(0, 9)
y = [19, 18.2, 20.2, 21.1, 22, 24.3, 23, 25.8, 24]

a, b, r_value, p_value, std_err = stats.linregress(x, y)

print ('r value', r_value)
print ('p_value', p_value)
print ('standard deviation', std_err)

line = a*x + b
plot(x, line, 'r-', x, y,'o')
show()