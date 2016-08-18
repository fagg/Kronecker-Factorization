function T = tilde(A, mb, nb)
[m,n] = size(A);
mc = m / mb;
nc = n/nb;

T = zeros(mb*nb, mc*nc);
x = zeros(1, mc*nc);

for ib = 1:mb
    for jb = 1:nb
        x(:) = A((ib-1)*mc+1:ib*mc, (jb-1)*nc+1:jb*nc);
        T((jb-1)*mb+ib,:) = x;
    end
end

end