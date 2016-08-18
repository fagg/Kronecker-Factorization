clear all, close all;

M = randn(1568, 6272);
Bsz = [49 49];
Csz = [32 128];

fprintf('Computing Van Loan-Pitsianis approximation...\n');
[A, B] = strongKronApprox(M, Bsz(1), Bsz(2), Csz(1), Csz(2));
Mvl = zeros(size(M));
for i = 1:numel(A)
    Mvl = Mvl + kron(A{i}, B{i});
end

fprintf('Nearest Kronecker Sum Estimate residual: %f\n', norm(Mvl(:)-M(:), 2)^2);
