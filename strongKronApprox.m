function [A, B, r] = strongKronApprox(C, mA, nA, mB, nB)
[m,n] = size(C);
n1 = m/(mA*mB);
n3 = n/(nA*nB);

Ch = zeros(n1*mA*nA, n3*mB*nB);
for i = 1:n1
    for j = 1:n3
       Ch((i-1)*mA*nA+1:i*mA*nA, (j-1)*mB*nB+1: j*mB*nB) = ...
           tilde(C((i-1)*mA*mB+1:i*mA*mB, (j-1)*nA*nB+1:j*nA*nB), mA, nA);
    end
end

r = rank(Ch);
A = cell(0);
B = cell(0);

[U,S,V] = svd(Ch);
s = sqrt(diag(S));
X = U(:,1:r)*diag(s(1:r));
Y = (V(:,1:r)*diag(s(1:r)))';

A0 = zeros(mA, nA);
B0 = zeros(mB, nB);

for i = 1:n1
   for j = 1:r
      A0(:) = X((i-1)*mA*nA+1:i*mA*nA, j);
      A{end+1} = A0;
   end
end

for i = 1:r
    for j = 1:n3
        B0(:) = Y(i,(j-1)*mB*nB+1:j*mB*nB);
        B{end+1} = B0;
    end
end


end