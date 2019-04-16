#!/usr/bin/env python
import numpy as np
from scipy import linalg
from scipy.sparse import diags
from numpy.linalg import inv
def main():
	
	#A	
	n = 10
	k = np.array([-1*np.ones(n-2),-1*np.ones(n-1),10*np.ones(n),-1*np.ones(n-1),-1*np.ones(n-2)])
	offset = [-2,-1,0,1,2]
	A = diags(k,offset).toarray()	
	#print(A)

	#D
	n = 10
	k = np.array([8*np.ones(n)])
	offset = [0]
	D = diags(k,offset).toarray()	
	#print(D)

	#M
	n = 10
	k = np.array([-1*np.ones(n-2),-1*np.ones(n-1),1*np.ones(n),0*np.ones(n-1),0*np.ones(n-2)])
	offset = [-2,-1,0,1,2]
	M = diags(k,offset).toarray()	
	#print(M)

	#N
	n = 10
	k = np.array([0*np.ones(n-2),0*np.ones(n-1),1*np.ones(n),-1*np.ones(n-1),-1*np.ones(n-2)])
	offset = [-2,-1,0,1,2]
	N = diags(k,offset).toarray()	
	#print(N)
	#print(M+N+D)
	#print(A)
#(M+D)x(k+1)=-Nx(k)+b
	m=M+D
	#print(m)
	m1=inv(np.matrix(m))
	#print(m1)
	n=np.dot(-1,N)
	main_m=np.dot(m1,n)
	##=print("main_m is:",main_m)
	eigenvalues = np.linalg.eigvals(main_m)
	#print("eigenvalues are:",eigenvalues)
	#iterations
	s=(10,1)
	b=np.ones(s)
	c=np.dot(m1,b)
	x=linalg.solve(A, b)
	print("the aswer is:",x)
	iteration=0
	x1 = np.ones((10,1))
	err = np.ones((10,1))*100
	#print(np.dot(m1,b))
	while np.max(err) > 1:
		iteration=iteration+1
		xn=np.dot(main_m,x1)+c
		err = abs((xn - x1)/(xn+0.00001))*100
		x1 = xn
	
	print("first:",x1)	
	print(iteration)
	
#Dx(k+1)= -(M+N)x(k)+b
	m1=M+N
	m2=np.dot(-1,m1)
	d=inv(np.matrix(D))
	main_m2=np.dot(d,m2)
	print("main_m2:",main_m2)
	eigenvalues2 = np.linalg.eigvals(main_m2)
	print("eigenvalues are:",eigenvalues2)
	#iteration
	iteration2=0
	x2 = np.ones((10,1))
	err2 = np.ones((10,1))*100
	while np.max(err2) > 0:
		iteration2=iteration2+1
		xn2 = np.dot(main_m2,x2)+np.dot(d,b)
		err2 = abs((xn2 - x2)/xn2)*100
		x2 = xn2
	print("second:",x2)	
	print(iteration2)

#(M+N)x(k+1)=-Dx(k)+b
	
	t=M+N

	m3=inv(np.matrix(t))

	r=np.dot(-1,D)
	main_m3=np.dot(m3,r)

	eigenvalues3 = np.linalg.eigvals(main_m3)
	print("eigenvalues are:",eigenvalues3)
	print("not converging")


if __name__ == '__main__':
	main()
