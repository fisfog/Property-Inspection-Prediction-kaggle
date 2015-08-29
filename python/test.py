import sys


n = 100000
for i in xrange(n):
	sys.stdout.write(str(i*100.0/n)+'%\r')
