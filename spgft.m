function spG=spgft(PT,G,rho,Cp,wt,PTc,mask,lam,mdrv,ordr,nreg,normflg)
% create a spline for Gibbs energy - use G (optionally rho and Cp) as constraint and
% regularize in the non-physical region with smooth mdrv th derivatives. 
%
%    The expectation is that units are MKS!! (J/kg, ,MPa, K kg/m^3)
% 
%    G rho and Cp are typicially grids of values with PT containing
%    cells with P and T values.  Alternative is to enter G along a base -
%    In that case G is a vector of values and Cp and rho are used to
%    reconstruct G at higher pressures
%
% function call:
%    sp_G=spgft(G,rho,Cp,wt,PT,PTc,mask,lam,mdrv,ordr,nreg,normflg)
%
%    The problem is that the spline is on a rectangular grid of points. In the real world, 
%    phase boundaries interfere and sections of the grid represent metastable regions where 
%    no reasonable thermodynamic values are available.  Thus, to solve the problem on the grid (insure that the
%    matrixes are adequately conditions and not too rank-deficit) regularization is applied 
%        - assert that the surfaces need to be "smooth" where data are not available. 
%    Input "mask" has ones where input thermodynamic properties are valid and NaN where they are not
%    input "PT" is a cell containing P and T
%    input "PTg" is a cell containing the control points for the spline
%               (empty to use the PT grid)
%    input "lam" is vector of smoothing parameters
%    input "wt" is vector of weights for the relative importance of densities and Cp
%    input "mdrv" is vector of the derivatives to use in smoothing
%    input "ordr" is vector of integer order's for the spline
%    input "nreg" sets the number of regularization points for each control point
%    input "normflg" =1 to rescale data based on their standard deviations, =0 to not rescale data
%
% custom functions mkCmn wt_area and mkgrid are nested in this file
%  JMB 2015

% extract P and T from cell, convert P to Pa units
P=PT{1}*1e6;
P=P(:);
T=PT{2};
T=T(:)';

% ditto for control points
if not(isempty(PTc))
    Pc=PTc{1}*1e6;  
    Tc=PTc{2};
else
    Pc=PT{1}*1e6;  
    Tc=PT{2}; 
end

% determine number of data and control points
nP=length(P);
nT=length(T);
nPc=length(Pc);
nTc=length(Tc);


if(isempty(mask)),mask=ones(nP,nT);end % default to using all data if no mask is provided

mdrv=mdrv+1; %increment to align chosen regularization derivative with the order of basis functions returned by spcol

% find indexes for regions where G is defined and regions where it is not
id_incl=find(not(isnan(mask(:))));
n_incld=length(id_incl);

% determine whether density or Cp data are included
if(isempty(rho)),rhoflg=0;else rhoflg=1;end
if(isempty(Cp)),Cpflg=0;else Cpflg=1;end
% determine if G is defined only at the base or everywhere
if min(size(G))==1,Gvflg=1;else Gvflg=0;end
% if the damping vector is empty - do calcuations with no regularization
if(not(isempty(lam))),reg_flg=1;else reg_flg=0;end

% need some protection against bad input.  This idea did not work so well and is commented out
%Gmax=max(max(abs(G(id_incl))));
%if Gmax>1e9; error('Check your matrix of values for G'),end  % don't proceed with bad values in G

m_rb=mask(1,:);
id_rb=find(not(isnan(m_rb(:))));
n_dat=length(id_incl);
norm_fac=sqrt(n_dat);  % will weight data vs regularization by square root of number of data points

if Gvflg % handle G only at 1 bar or G for all P and T
    G_norm=std(G(id_rb))^-1; % use standard deviation of G at 1 bar for normalization
else
    G_norm=std(G(id_incl(:)))^-1; % use standard deviation of G(P,T) for normalization
end

% rho and Cp expressed as linear derivatives of G and determine standard
% deviations of the "data" derivatives
if rhoflg, Gp=rho.^-1;Gp_norm=std(Gp(id_incl(:)))^-1;end    % use standard deviation of Gp for normalization
if Cpflg,Gtt=-Cp./(ones(nP,1)*T);Gtt_norm=std(Gtt(id_incl(:)))^-1;end   % use standard deviation of Gtt for normalization

% define the knot sequence and co-locate the data default of 6tyh order splines
if isempty(ordr)
    kt=6;
    kp=6;
else
    kp=ordr(1);
    kt=ordr(2);
end

Pknts=aptknt(Pc(:)',kp);
Tknts=aptknt(Tc(:)',kt);
Tcol = spcol(Tknts,kt,brk2knt(T,kt));
Pcol = spcol(Pknts,kp,brk2knt(P,kp));
 
% create the matrixes of basis functions
% depends on whether G is defined everywhere or just at the base
if Gvflg
    RdatG=makeCmn(Pcol(1,:),Tcol(1:kt:end,:));
else
    RdatG=makeCmn(Pcol(1:kp:end,:),Tcol(1:kt:end,:));
end
% basis functions for densities (volumes) and specific heats:
RdatV=makeCmn(Pcol(2:kp:end,:),Tcol(1:kt:end,:));
RdatC=makeCmn(Pcol(1:kp:end,:),Tcol(3:kt:end,:));

% in the case that regularization is used, set up the basis functions
% then weight by volume contained around each regularization point
if reg_flg
    Pr=mkgrid(Pc,nreg);
    Tr=mkgrid(Tc,nreg);
    Tcolr = spcol(Tknts,kt,brk2knt(Tr,kt));
    Pcolr = spcol(Pknts,kp,brk2knt(Pr,kp));
    wtreg=area_wt({Pr,Tr});
    RregT=makeCmn(Pcolr(1:kp:end,:),Tcolr(mdrv(2):kt:end,:));    
    RregP=makeCmn(Pcolr(mdrv(1):kp:end,:),Tcolr(1:kt:end,:));    
    [nr,nc]=size(RregT);
    [it,jt,xt]=find(RregT);    
    [ip,jp,xp]=find(RregP);
    RregT=sparse(it,jt,wtreg(it).*xt,nr,nc);      
    RregP=sparse(ip,jp,wtreg(ip).*xp,nr,nc);
end

% set up the spline structure for the output
spG.form='B-';
spG.knots={Pknts,Tknts};
spG.number=[nPc nTc];
spG.order=[kp kt];
spG.dim=1;

% the  normalization for G, Gp and Gtt should help. Experience is that it
% does not work as well as expected,  the input normflg allows this feature to be turned off or on 
% if(isnan(G_norm) | isinf(G_norm) ), error('bad values in G'),end
% if(isnan(Gp_norm) | isinf(Gp_norm) ), error('bad values in density'),end
% if(isnan(Gtt_norm) | isinf(Gtt_norm) ), error('bad values in Cp'),end

if not(normflg)
   G_norm=1;
   Gp_norm=1;
   Gtt_norm=1;
end

% set up matrixes for solution of coef=A\B
% a number of different setups depending on all the choices for "data" and regularization
if reg_flg
    if (rhoflg && Cpflg)
        if Gvflg
            A=[G_norm*RdatG(id_rb,:);wt(1)*Gp_norm*RdatV(id_incl,:); wt(2)*Gtt_norm*RdatC(id_incl,:); norm_fac*lam(1)*RregP;norm_fac*lam(2)*RregT];
            B=[G_norm*G(id_rb(:))';wt(1)*Gp_norm*Gp(id_incl(:));wt(2)*Gtt_norm*Gtt(id_incl(:));zeros(2*nr,1)];
        else
            A=[G_norm*RdatG(id_incl,:);wt(1)*Gp_norm*RdatV(id_incl,:); wt(2)*Gtt_norm*RdatC(id_incl,:); norm_fac*lam(1)*RregP;norm_fac*lam(2)*RregT];
            B=[G_norm*G(id_incl);wt(1)*Gp_norm*Gp(id_incl(:));wt(2)*Gtt_norm*Gtt(id_incl(:));zeros(2*nr,1)];
        end
    elseif (rhoflg && not(Cpflg))
        A=[G_norm*RdatG(id_incl,:);wt(1)*Gp_norm*RdatV(id_incl,:); norm_fac*lam(1)*RregP;norm_fac*lam(2)*RregT];
        B=[G_norm*G(id_incl);wt(1)*Gp_norm*Gp(id_incl(:));zeros(2*nr,1)];      
    elseif (not(rhoflg) && Cpflg)
        A=[G_norm*RdatG(id_incl,:); wt(2)*Gtt_norm*RdatC(id_incl,:); norm_fac*lam(1)*RregP;norm_fac*lam(2)*RregT];
        B=[G_norm*G(id_incl);wt(2)*Gtt_norm*Gtt(id_incl(:));zeros(2*nr,1)];        
    elseif (not(rhoflg) && not(Cpflg))
        A=[G_norm*RdatG(id_incl,:);norm_fac*lam(1)*RregP;norm_fac*lam(2)*RregT];
        B=[G_norm*G(id_incl);zeros(2*nr,1)];      
    end
else
    if (rhoflg && Cpflg)
        A=[G_norm*RdatG(id_incl,:);wt(1)*Gp_norm*RdatV(id_incl,:); wt(2)*RdatC(id_incl,:)];
        B=[G_norm*G(id_incl);wt(1)*Gp_norm*Gp(id_incl(:));wt(2)*Gtt(id_incl(:))];
    elseif (rhoflg && not(Cpflg))
        A=[G_norm*RdatG(id_incl,:);wt(1)*Gp_norm*RdatV(id_incl,:)];
        B=[G_norm*G(id_incl);wt(1)*Gp_norm*Gp(id_incl(:))];        
    elseif (not(rhoflg) && Cpflg)
        A=[G_norm*RdatG(id_incl,:);wt(2)*Gtt_norm*RdatC(id_incl,:)];
        B=[G_norm*G(id_incl);wt(2)*Gtt_norm*Gtt(id_incl(:))];
    elseif (not(rhoflg) && not(Cpflg))
        A=G_norm*RdatG(id_incl,:);
        B=G_norm*G(id_incl);
    end
end

spG.coefs=reshape((A\B),nPc,nTc); % solve normal equations
% alternative solutions: (try both compare - typically does not change the results)
%spG.coefs=reshape(((A'*A)\(A'*B))',nPc,nTc);  % least squares solution

spG.knots{1}=spG.knots{1}/1e6;  % convert back to MPa

% end of G spline creation

function D=makeCmn(A,B,C)
% This function combines collocation matrixes A(i,j) B(k,l) C(p,q) for 
% three dimensions (associated with a 4D hypersurface tensor spline)
% into a single matrix of size i*k*p by j*l*q. The idea is that the 
% tensor spline calculation is given as:
%    sum sum sum (ABC*a)  where A,B, and C are collocation matrix  and a is
%    an array of coefficients. 
% This can be remapped into a single matrix multiply D*a(:)
%    where D is the matrix returned here.
%  If working in 3D pass only matrixes A and B to this function
%  If A and B are row vectors, makeCmn returns a
%  single row corresponding to a single "data" point.
%
% This is mostly a problem of keeping track of how matrixes are indexed. 
% Below is the algorithm using a full set of for loops
% [ai,aj]=size(A);
% [bk,bl]=size(B);
% [cp,cq]=size(C);
% for p=1:cp
%   for k=1:bk      
%     for i=1:ai
%          mm= i + (k-1)*ai + bk*ai*(p-1);
%          a=A(i,:);  %one loop over an index can be vectorized
%          for q=1:cq  
%            for l=1:bl
%                 nnc=(l-1)*aj + bl*aj*(q-1);
%                 D(mm,nnc+(1:aj))=a*B(k,l)*C(p,q);
%            end
%          end
%      end
%   end
% end
% % % 
% JMB 2011

% Below I take advantage of sparseness - 
[ai,aj]=size(A);
[bk,bl]=size(B);
%Determine whether there is another dimension to the problem
switch nargin
    case 3
     [cp,cq]=size(C);
    case 2
     C=1;
     cp=1;cq=1;
end

%Determine (1) which elements are non-zero (A,B,C are sparse) and (2) the sizes of the matrixes
nR= ai*bk*cp;  
nC= aj*bl*cq;
[ii,jj,sa]=find(A);
[kk,ll,sb]=find(B);
[pp,qq,sc]=find(C);
ni=length(ii);
nk=length(kk);
np=length(pp);
nmm=ni*nk*np;

% preallocate vectors
mm=zeros(nmm,1);
nn=zeros(nmm,1);
v=zeros(nmm,1);

%Do the work
for p=1:np
 for k=1:nk
     im= (kk(k)-1)*ai + bk*ai*(pp(p)-1);
     in= (ll(k)-1)*aj + bl*aj*(qq(p)-1);
     count=(k-1)*ni +ni*nk*(p-1);
     for i=1:ni 
       mm(count+i)= ii(i) + im;  
       nn(count+i)= jj(i) + in;
       v(count+i)=sa(i)*sb(k)*sc(p);
     end
 end
end

%Assemble the resultant sparse matrix
D=sparse(mm,nn,v,nR,nC);


function wt=area_wt(PT)
% Usage: wt=area_wt(PT) where PT is a cell containting the P and T grid points
% this subfunction assigns a weight to every regularization point based on
% the square root of the volume encompassed by that point
%  ie  sum of wt^2 is 1
if(iscell(PT))
 if (length(PT)==2)   
    p=PT{1};
    dp=diff(p(:))';
    t=PT{2};
    dt=diff(t(:))';
    wtp=.5*[dp(1) dp(1:end-1)+dp(2:end) dp(end)];
    wtt=.5*[dt(1) dt(1:end-1)+dt(2:end) dt(end)];
    wt=wtp(:)*wtt;
    wtsum=sum(wt(:));
    wt=sqrt(wt/wtsum);
 elseif( length(PT)==3)  
      p=PT{1};
      dp=diff(p(:))';
      t=PT{2};
      dt=diff(t(:))';
      m=PT{3};
      dm=diff(m(:))';
    wtp=.5*[dp(1) dp(1:end-1)+dp(2:end) dp(end)];
    wtt=.5*[dt(1) dt(1:end-1)+dt(2:end) dt(end)];
    wtm=.5*[dm(1) dm(1:end-1)+dm(2:end) dm(end)];
    [a,b,c]=ndgrid(wtp,wtt,wtm);
    wt=a.*b.*c;
    wtsum=sum(wt(:));
    wt=sqrt(wt/wtsum);
  end
 else    
    dx=diff(PT(:))';
    wtx=.5*[dx(1) dx(1:end-1)+dx(2:end) dx(end)];
    wtsum=sum(wtx(:));
    wt=sqrt(wtx/wtsum);
end
 
function xg=mkgrid(x,nreg)
% this subfunction puts nreg regularization points between every control point
xg=x(:)';
dx=diff(xg);
if nreg>1;
for i=2:nreg
    xg=[xg x(1:end-1)+(i-1)*dx/nreg];
end
end
xg=sort(xg);
