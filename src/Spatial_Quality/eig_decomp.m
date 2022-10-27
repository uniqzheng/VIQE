function [eigvec_St1,eigval_St1]=eig_decomp(St1);
[eigvec,eigval]=eig(St1);
eigval=abs(diag(eigval)');		
[eigval,I]=sort(eigval);
eigval_St1=fliplr(eigval); 
eigvec_St1=fliplr(eigvec(:,I));