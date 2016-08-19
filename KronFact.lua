t = require 'torch'
c = require 'cutorch'
m = require 'math'

function kron(A,B)
   local m, n = A:size(1), A:size(2)
   local p, q = B:size(1), B:size(2)
   local C = torch.DoubleTensor(m*p,n*q)

   for i=1,m do
      for j=1,n do
         C[{{(i-1)*p+1,i*p},{(j-1)*q+1,j*q}}] = torch.mul(B, A[i][j])
      end
   end
   return C
end


function tilde(A, mb, nb)
   local m = A:size(1)
   local n = A:size(2)

   local mc = m / mb
   local nc = n / nb

   local T = torch.DoubleTensor(mb*nb, mc*nc):zero()
   local x = torch.DoubleTensor(1, mc*nc):zero()


   for ib = 1, mb do
      for jb = 1, nb do
         local rowSelect = torch.range((ib-1)*mc+1, ib*mc, 1):long()
         local colSelect = torch.range((jb-1)*nc+1, jb*nc, 1):long()
         local Asel = A:index(1, rowSelect):index(2, colSelect)
         x = torch.reshape(Asel, 1, mc*nc)
         T[{{(jb-1)*mb+ib}, {}}] = x
      end
   end


   return T
end

function kronApprox(C, mA, nA, mB, nB)
   local m = C:size(1)
   local n = C:size(2)
   local n1 = m/(mA*mB)
   local n3 = n/(nA*nB)

   local Ch = torch.DoubleTensor(n1*mA*nA, n3*mB*nB)

   for i = 1, n1 do
      for j = 1, n3 do
         local ChrIdx = torch.range((i-1)*mA*nA+1, i*mA*nA, 1)
         local ChcIdx = torch.range((j-1)*mB*nB+1, j*mB*nB, 1)
         local CrIdx = torch.range((i-1)*mA*mB+1, i*mA*mB, 1):long()
         local CcIdx = torch.range((j-1)*nA*nB+1, j*nA*nB, 1):long()

         local CSel = C:index(1, CrIdx):index(2, CcIdx)
         local CSelTil = tilde(CSel, mA, nA)
         Ch[{{ChrIdx}, {ChcIdx}}] = CSelTil
      end
   end

   U,S,V = torch.svd(Ch)
   local s = torch.sqrt(S)
   local r = s:nonzero():size(1) -- this is the rank
   A = {}
   B = {}

   local sVW = torch.diag(s:index(1, torch.range(1,r,1):long()))
   print(sVW)
   print(U:size())
   local X = U:index(2, torch.range(1,r,1):long()) * sVW
   print(X:size())
   local Y = (V:index(2, torch.range(1,r,1):long()) * sVW):t()
   print(Y:size())


   local A0 = torch.DoubleTensor(mA, nA)
   local B0 = torch.DoubleTensor(mB, nB)

   local pp = 1
   for i = 1, n1 do
      for j = 1, r do
         local rowSelect = torch.range((i-1)*mA*nA+1, i*mA*nA, 1):long();
         local colSelect = torch.LongTensor{j}
         
         local A0sel = X:index(1, rowSelect):index(2, colSelect)
         A0 = torch.reshape(A0sel, mA, nA):t() -- because row major
         A[pp] = A0
         pp = pp + 1
      end
   end

   pp = 1

   for i = 1, r do
      for j = 1, n3 do
         local rowSelect = torch.LongTensor{i};
         local colSelect = torch.range((j-1)*mB*nB+1, j*mB*nB):long()
         local B0sel = Y:index(1, rowSelect):index(2, colSelect)
         B0 = torch.reshape(B0sel, mB, nB)
         B[pp] = B0
         pp = pp + 1
      end
      
   end

   return A,B
   
end



-----------------------------------------
M = torch.randn(49*32, 49*128)
A,B = kronApprox(M, 49, 49, 32, 128)

Mvl = torch.zeros(49*32, 49*128)

print('Forming estimate...')
for i = 1, table.getn(A) do
   Mvl:add(kron(A[i], B[i]):double())
end

local residual = torch.norm(Mvl-M) * torch.norm(Mvl-M)
print('Error: ' .. residual)

