function [T,V,Var,kurt]=SPPA(X,varargin)
%
% Modification of the original version.
%   o Added improved selection of initial population to ensure maximum
%     coverage of variables. If population size is sufficient, each
%     variable is selected a minimum of n times. Residual individuals are
%     selected at random without repetition. This is more equitable than
%     the original version which might exclude some variables and
%     over-represent others.
%
% SPPA sparse projection pursuit analysis using genetic algorithm.
%
% Outputs:
%
% T = SPPA(X) performs sparse projection pursuit on the m*n (samples x variables)
%       matrix X, and returns the scores in T, a m*dim matrix, where dim is
%       the number of dimensions of separation.
%
% [T,V] = SPPA(X) also returns the vectors in V, a dim*n matrix containing the
%       projection vectors for each dimension.
%
% [T,V,VAR] = SPPA(X) returns the variables in VAR, a  dim*nvars matrix, with each
%       column containing the chosen variables for each dimension of separation.
%
% [T,V,VAR,KURT] = SPPA(X) also returns the kurtosis value(s) for the
%       solution, in a dim*1 matrix, with each dimension having a kurtosis
%       value associated with the separation.
% Inputs:
%
% SPPA(X,...,Optionname,Optionvalue,...) performs sparse projection
%       pursuit according to the options provided. Optionname is a string,
%       as identified below, and Optionvalue is either a scalar or a string,
%       to identify the value.
%
% Options:
%
%    Dim: Number of dimensions along which separation is desired (1-3)
%           Default: 2
%           Recommendation: 1 dim for 2 classes, 2 dim for 3-4 classes,
%                           3 dim for 5-8 classes
%    Nvars: Number of variables
%           Default: 5
%           Recommendation: Nvars ~= nsamp/25, such that nvars >= 3
%    Mutrate: Mutation rate (decimal between 0 and 1)
%           Default: 0.1
%    Popsize: Population size
%           Default: 100
%    Meth: Univariate (uni) or multivariate (mul) kurtosis
%           Default: uni
%    Opt: Ordinary (ord) or recentered (rec) kurtosis
%           Default: ord
%    Maxtime: Maximum time (in seconds), after which the algorithm stops if
%           it hasn't converged
%           Default: 300
%    Ctoff: Fitness value below which all individuals are considered
%           equivalent in mating, because they give adequate separation.
%           Default: 1.5 (univariate), 4.5 (multivariate), 2.25 (prod),
%           3 (sum).
%
%
% Example: To perform sparse projection pursuit with 10 variables,
%  using recentered kurtosis and a mutation rate of 20%, enter:
%  [T,V,VAR] = SPPA(X,'nvars',10,'opt','rec','mutrate',0.2)

% Demo: The script file demo.m (SI of paper and also found at
%  https://github.com/S-Driscoll/SparseProjectionPursuit) can be run to
%  to show SPPA in action on a real data set (Salmon NMR data).
%

p=inputParser;
addOptional(p,'objclass',[])
addOptional(p,'classlist',{},@iscell)
addParameter(p,'dim',2,@(x) assert(0<x && x<4));
addParameter(p,'nvars',5,@(x) assert(x>1));
addParameter(p,'mutrate',0.1,@(x) assert(0<=x && x<1));
addParameter(p,'popsize',100,@(x) assert(x>0));
addParameter(p,'opt','ord',@(x) assert(isscalar(strfind('ordrec',x))));
addParameter(p,'meth','uni',@(x) assert(isscalar(strfind('unimul',x))));
addParameter(p,'maxtime',300,@(x) assert(x>0));
addParameter(p,'pctrecomb',0.3,@(x) assert(x>0));
addParameter(p,'sumprod','norm',@(x) assert(isscalar(strfind('normsumprod',x))));
addParameter(p,'exponent',4,@(x) assert(x>0));
addParameter(p,'ctoff',1.5,@(x) assert(x>0));
addParameter(p,'stat',50,@(x) assert(x>0)); %Definition of a static population (how many generations)
parse(p,varargin{:})
f=p.Results;
if strcmp(f.meth,'mul') && strcmp(f.sumprod,'norm')==false
    error('Multivariate kurtosis is incompatible with sum/product options')
end
if strcmp(f.opt,'rec') && strcmp(f.sumprod,'norm')==false
    warning('Use of recentered kurtosis is not recommended with sum/product options')
end
if strcmp(f.sumprod,'norm')
    [T,V,Var,kurt,pop]=ppga2(X,varargin{:});
else
    [T,V,Var,kurt]=ppga6(X,varargin{:});
end
end
%% -------------Dimension-by-dimension separation algorithm-------------------%%
function [scores,vectors,variables,kurt,pops]=ppga2(X,varargin)
%PPGA2  Variable selection for projection pursuit analysis, with a genetic
%       algorithm (version 2.0) with dimension-by-dimension separation.

p=inputParser;
addOptional(p,'objclass',[])
addOptional(p,'classlist',{},@iscell)
addParameter(p,'dim',2,@isnumeric)
addParameter(p,'nvars',5);
addParameter(p,'maxgen',1000,@(x) assert(x>0));
addParameter(p,'maxtime',300,@(x) assert(x>0));
addParameter(p,'meth','uni',@(x) assert(isscalar(strfind('unimul',x)))); %Univariate vs multivariate kurtosis
addParameter(p,'opt','ord',@(x) assert(isscalar(strfind('ordrec',x)))); %Ordinary vs recentered kurtosis
addParameter(p,'mutrate',0.1,@(x) assert(0<=x && x<1));
addParameter(p,'popsize',100,@(x) assert(x>0));
addParameter(p,'pctrecomb',0.3,@(x) assert(x>0));
addParameter(p,'stat',50,@(x) assert(x>0)); %Definition of a static population (how many generations)
addParameter(p,'exponent',4,@(x) assert(x>0));
addParameter(p,'ctoff',1.5,@(x) assert(x>0));
addParameter(p,'sumprod','norm',@(x) assert(isscalar(strfind('normsumprod',x))));
parse(p,varargin{:})
f=p.Results;
%Determine how many sets of variables it will output (dimvar)
%Determine how many dimensions to ask from the projection
%   pursuit alg (pursuitdim)
if strcmp('mul',f.meth)
    dimvar=1;
    pursuitdim=f.dim;
    f.ctoff=4.5;
else
    dimvar=f.dim;
    pursuitdim=1;
end

[nsamp,totvars]=size(X); %total number of variables
scores=zeros(nsamp,f.dim); %Will hold pp scores
variables=zeros(dimvar,f.nvars); %Will hold chosen variables
vectors=zeros(totvars,f.dim);
if rem(f.popsize,2) %Set number of elite individuals
    numret=1;
else
    numret=2;
end
nchild=f.popsize-numret; %the remaining population is made of retained individuals
%Mean center the data:
Morig=ones(nsamp,1)*mean(X);
X=X-Morig;
X0=X;
tic
for d=1:f.dim
    %     disp(['Dimension ' num2str(d)])
    %% Define default variables
    pop=zeros(f.popsize,f.nvars);
    populations=zeros(f.popsize,f.nvars,f.maxgen); %3d matrix to store all of the historical populations
    fitness=zeros(f.popsize,f.maxgen); %Matrix storing fitness of all previous individuals
    
    %% Construct initial population
    % Modified code for initial population.  Designed to ensure balanced
    % representation of all variables. If popsize*nvar>=totvars, population is
    % divided into n3 groups, where each variable is represented at least once.
    % Residual number (n4) are filled randomly without repetition. This ensures
    % maximum, unbiased coverage of variables.
    %
    n1=ceil(totvars/f.nvars);           % Number for full group
    n2=rem(totvars,f.nvars);            % Number in last cell
    n3=floor(f.popsize/n1);             % Number of full groups
    n4=rem(f.popsize,n1);               % Number left over
    for igrp=1:n3              % Fills in each group
        temp=1;                % Note: last member may need padding
        while temp~=0          % This ensures no repetition in last member
            indx1=[randperm(totvars) randperm(totvars,rem(f.nvars-n2,f.nvars))];
            temp=f.nvars-length(unique(indx1(end-f.nvars+1:end)));
        end
        indx1=reshape(indx1,f.nvars,n1)'; % indx1 contains all variables
        indx2=(igrp-1)*n1;                % Starting index to add to pop
        for jj=1:n1                       % Add to population
            indx3=indx2+jj;
            pop(indx3,:)=indx1(jj,:);
            [~,~,PPOUT]=projpursuit(X(:,pop(indx3,:)),pursuitdim,1,f.meth,f.opt);
            fitness(indx3,1)=PPOUT.K;
        end
    end
    indx2=n3*n1;                         % This part adds residual to pop
    indx1=randperm(totvars,n4*f.nvars);  % Generate residual members
    indx1=reshape(indx1,f.nvars,n4)';
    for jj=1:n4                          % Add to pop
        indx3=indx2+jj;
        pop(indx3,:)=indx1(jj,:);
        [~,~,PPOUT]=projpursuit(X(:,pop(indx3,:)),pursuitdim,1,f.meth,f.opt);
        fitness(indx3,1)=PPOUT.K;
    end
    [fitness(:,1),isort]=sort(fitness(:,1)); %sort individuals by fitness
    pop=pop(isort,:);
    
    %%
    populations(:,:,1)=pop; %Store first generation
    for k=2:f.maxgen
        %% Elite selection
        popelite=pop(1:numret,:);
        %% Mating
        popchild=mating7(f.nvars,pop,nchild,fitness(:,k-1),popelite,totvars,f.pctrecomb,f.ctoff,f.exponent);
        %% Random mutation
        pop=mutation(nchild,f.mutrate,f.nvars,totvars,popelite,popchild);
        %% Evaluate fitness of each member
        %         disp(['Generation ', num2str(k)])
        fitness2=fitness;
        for i=1:f.popsize
            [~,~,PPOUT]=projpursuit(X(:,pop(i,:)),pursuitdim,1,f.meth,f.opt);
            fitness(i,k)=PPOUT.K;
            %% Check for better fitness previously
            r=1;
            while r<51 && r<k %Check last 50 generations in sequence
                loctf=ismember(populations(:,:,k-r),pop(i,:),'rows'); %Find matches
                if sum(loctf)>0
                    loc=find((loctf==true),1);
                    fitness(i,k)=min(fitness(i,k),fitness2(loc,k-r));%Replace
                    break
                end
                r=r+1;
            end
        end
        [fitness(:,k),isort]=sort(fitness(:,k)); %sort individuals by fitness
        %% Intermediate plotting of fitness
        if k==2
            figure(50)
            clf
            drawnow
        end
        if k>2
            delete(h1);
            delete(h2);
        end
        plot(median(fitness(:,1:k)),'r','LineWidth',2.5)
        hold on
        plot(min(fitness(:,1:k)),'b','LineWidth',2.5)
        grid on
        axis tight
        xlabel('Generation number')
        ylabel('Kurtosis')
        h1=annotation('textbox',[.6 0.7 .3 .2],'String',strcat('Median Kurtosis: ',num2str(median(fitness(:,k)))),'EdgeColor','none','Color','red');
        h2=annotation('textbox',[.6 0.65 .3 .2],'String',strcat('Minimum Kurtosis: ',num2str(min(fitness(:,k)))),'EdgeColor','none','Color','blue');
        drawnow
        oldpop=pop;
        for i=1:f.popsize
            pop(i,:)=oldpop(isort(i),:);
        end
        
        %% Store population
        populations(:,:,k)=pop;
        %% Test convergence
        if k==f.maxgen
            disp(['Failed to converge after ' num2str(k) ' generations'])
            break
        end
        if k>f.stat && isequal((populations(1,:,k)),(populations(1,:,k-f.stat)))
            disp(['Stopping due to static population (' num2str(k) ' generations)'])
            break
        end
        if toc>f.maxtime*d
            disp(['Maximum time reached (' num2str(f.maxtime) ' seconds)'])
            break
        end
    end
    
    %% Evaluate class separation
    figure(50)
    hold off
    variables(d,:)=sort(pop(1,:));
    if strcmp('mul',f.meth)==true
        [scores,V_1,PPOUT]=projpursuit(X(:,variables(d,:)),pursuitdim,'mul');
        V2=zeros(totvars,f.dim);
        V2(variables,:)=V_1;
        vectors=V2;
        kurt=PPOUT.K;
        pops=pop;
        break
    end
    [T_1,V_1,PPOUT]=projpursuit(X(:,variables(d,:)),1,f.opt);%Re-run with 100 guesses
    kurt(d)=PPOUT.K; %Store kurtosis
    V2=zeros(totvars,1);
    V2(variables(d,:))=V_1;
    scores(:,d)=T_1 + (ones(nsamp,1)*mean(X)*V2) - (ones(nsamp,1)*mean(X0)*V2) ;
    vectors(:,d)=V2'; %Prepare vectors for output
    pops(:,:,d)=pop;
    t=X*V2;
    T1(:,d)=t;
    P(:,d)=X'*t/(t'*t);
    if d<f.dim %Deflation
        X=X0-(T1*P');
        if d==2
            t=(scores(:,1)).*(scores(:,2));
            T1(:,3)=t;
            P(:,3)=X'*t/(t'*t);
            X=X0-(T1*P');
        end
    end
end
if strcmp('mul',f.meth)==false
    V=vectors*inv(P'*vectors);
    vectors=V;
    scores=X0*V;
end
toc
end

%% -------------------Mutation function--------------------------%%
function [pop]=mutation(nchild,mutrate,nvars,totvars,popelite,popchild)
for i=1:nchild
    while sum(ismember(sort([popelite;popchild],2),sort(popchild(i,:),2),'rows'))>1 %No duplicate individuals
        mutloc=rand(1,nvars)/mutrate; %generate a matrix saying where to mutate, for each child
        loc=find(mutloc<1);
        for j=1:length(loc)
            popchild(i,loc(j))=randi(totvars,1); %Replace with a random variable
            while sum(popchild(i,loc(j))==popchild(i,:))>1 %No duplicate variables
                popchild(i,loc(j))=randi(totvars,1);
            end
        end
    end
end
pop=[popelite;popchild]; %Concatenate elite and children
end

%% ------------Fitness based mating for dim by dim--------------------------%%
function [popchild]=mating7(nvars,pop,nchild,fitness,newpop,totvars,pctrecomb,ctoff,exponent)
popchild=zeros(nchild,nvars);
fitness(fitness<ctoff)=ctoff;
y=1./fitness.^exponent;
ranks=cumsum(y)/sum(y); %Generate cummulative distribution function
for i=1:2:nchild-1  %Go 2 children at a time
    it=0;
    parents=zeros(2,nvars);
    while (any(diff(sort(popchild(i,:)))==0)||any(diff(sort(popchild(i+1,:)))==0)) ... % No duplicate variables
            || sum(ismember(sort([newpop;popchild],2),sort(popchild(i,:),2),'rows'))>1 ... % No duplicate individuals
            || sum(ismember(sort([newpop;popchild],2),sort(popchild(i+1,:),2),'rows'))>1
        while parents(1,:)==parents(2,:) %Ensure no duplicate parents
            parents(1,:)=pop(find(ranks>=rand(1),1),:); %Choose parents
            parents(2,:)=pop(find(ranks>=rand(1),1),:);
        end
        crossover=rand(1,nvars)/pctrecomb;
        loc1=crossover<1;
        loc2=randperm(nvars,sum(loc1));
        popchild([i,i+1],:)=parents;
        popchild(i,loc1)=parents(2,loc2);
        popchild(i+1,loc2)=parents(1,loc1);
        it=it+1;
        if it>30 %Give up and make random individuals
            popchild(i,:)=randperm(totvars,nvars);
            popchild(i+1,:)=randperm(totvars,nvars);
        end
    end
end
end

%% -------------Sum/product separation algorithm-------------------------------%%
function [scores,vectors,variables,kurt]=ppga6(X,varargin)
%PPGA6  Variable selection for projection pursuit analysis, with a genetic
%       algorithm (version 6.0), simultaneous optimization

p=inputParser;
addOptional(p,'objclass',[])
addOptional(p,'classlist',{},@iscell)
addParameter(p,'dim',2,@(x) assert(x>0 && x<4))
addParameter(p,'nvars',5);
addParameter(p,'maxgen',1000,@(x) assert(x>0));
addParameter(p,'maxtime',300,@(x) assert(x>0));
addParameter(p,'opt','ord',@(x) assert(isscalar(strfind('ordrec',x)))); %Ordinary vs recentered kurtosis
addParameter(p,'mutrate',0.1,@(x) assert(0<=x && x<1));
addParameter(p,'popsize',100,@(x) assert(x>0));
addParameter(p,'sumprod','prod',@(x) assert(isscalar(strfind('sumprod',x))));
addParameter(p,'stat',50,@(x) assert(x>0)); %Definition of a static population
addParameter(p,'exponent',4,@(x) assert(x>0));
addParameter(p,'ctoff',2.25,@(x) assert(x>0));
parse(p,varargin{:})
f=p.Results;
f.sumprod=str2func(f.sumprod);
%Determine how many sets of variables it will output (dimvar)
scores=zeros(size(X,1),f.dim); %Will hold pp scores
variables=cell(1,f.dim); %Will hold chosen variables
[nsamp,totvars]=size(X); %total number of variables
vectors=zeros(totvars,f.dim);
kurt=zeros(f.dim,1);
if rem(f.popsize,2) %Set number of elite individuals
    numret=1;
else numret=2;
end
if strcmp('sum',f.sumprod)
    f.ctoff=3;
else f.ctoff=2.25;
end
nchild=f.popsize-numret; %the remaining population is made of retained individuals
%Mean center the data:
X=X-ones(nsamp,1)*mean(X);
X0=X; %Store initial X (because X is modified with deflation for each individual)
tic

%% Define default variables
pop=zeros(f.popsize,f.nvars*f.dim);
populations=zeros(f.popsize,f.nvars*f.dim,f.maxgen); %3d matrix to store all of the historical populations
fitness=zeros(f.popsize,f.maxgen); %Matrix storing fitness of all previous individuals

%% Construct initial population
%     disp('Generation 1')
for i=1:f.popsize
    T=zeros(nsamp,f.dim); %Initialize
    T1=zeros(nsamp,1);
    P=zeros(totvars,1);
    tempkurt1=zeros(f.dim,1);
    X=X0;
    for d=1:f.dim
        pop(i,((1+f.nvars*(d-1)):f.nvars*d))=randperm(totvars,f.nvars); %Create random pop
        [T(:,d),V,PPOUT]=projpursuit(X(:,pop(i,((1+f.nvars*(d-1)):f.nvars*d))),1,1,f.opt);
        tempkurt1(d)=PPOUT.K;
        V2=zeros(totvars,1);
        V2(pop(i,((1+f.nvars*(d-1)):f.nvars*d)))=V;
        if d<f.dim %Deflation
            t=X*V2;
            T1(:,d)=t;
            P(:,d)=X'*t/(t'*t);
            X=X0-(T1*P');
            if d==2 %Special deflation step to push for 8 groups in 3rd dim
                t=(T(:,1)).*(T(:,2));
                T1(:,3)=t;
                P(:,3)=X'*t/(t'*t);
                X=X0-(T1*P');
            end
        end
    end
    fitness(i,1)=f.sumprod(tempkurt1); %Calculate fitness based on sum or prod, as chosen
end
[fitness(:,1),isort]=sort(fitness(:,1)); %sort individuals by fitness
for i=1:f.popsize
    pop(i,:)=pop(isort(i),:);
end

%%
populations(:,:,1)=pop; %Store first generation
for k=2:f.maxgen
    %% Selection of elites
    popelite=pop(1:numret,:);
    %% Mating
    popchild=mating6(f.nvars,pop,nchild,fitness(:,k-1),popelite,totvars,f.dim,f.ctoff,f.exponent);
    %% Random mutation
    pop=mutation(nchild,f.mutrate,f.nvars*f.dim,totvars,popelite,popchild);
    %% Evaluate fitness of each member
    %         disp(['Generation ', num2str(k)])
    fitness2=fitness; %Necessary if fitness is evaluated in parallel
    for i=1:f.popsize
        X=X0; %Reset X for every individual
        tempkurt=zeros(f.dim,1);
        T=zeros(nsamp,f.dim);
        T1=zeros(nsamp,1);
        P=zeros(totvars,1);
        for d=1:f.dim
            [T(:,d),V,PPOUT]=projpursuit(X(:,pop(i,(1+f.nvars*(d-1)):f.nvars*d)),1,1,f.opt);
            tempkurt(d)=PPOUT.K;
            V2=zeros(totvars,1);
            V2(pop(i,((1+f.nvars*(d-1)):f.nvars*d)))=V;
            if d<f.dim %Deflation
                t=X*V2;
                T1(:,d)=t;
                P(:,d)=X'*t/(t'*t);
                X=X0-(T1*P');
                if d==2
                    t=(T(:,1)).*(T(:,2));
                    T1(:,3)=t;
                    P(:,3)=X'*t/(t'*t);
                    X=X0-(T1*P');
                end
            end
        end
        fitness(i,k)=f.sumprod(tempkurt); %Calculate fitness based on sum/prod
        %% Check for better fitness previously
        r=1;
        while r<51 && r<k %Check last 50 generations in sequence
            loctf=ismember(populations(:,:,k-r),pop(i,:),'rows'); %Find matches
            if sum(loctf)>0 %If found
                loc=find((loctf==true),1); %Where are they
                fitness(i,k)=min(fitness(i,k),fitness2(loc,k-r));%Replace
                break
            end
            r=r+1;
        end
    end
    [fitness(:,k),isort]=sort(fitness(:,k)); %sort individuals by fitness
    %% Intermediate plotting of fitness
    if k==2
        figure(50)
        clf
        drawnow
    end
    plot(median(fitness(:,1:k)),'r')
    hold on
    plot(min(fitness(:,1:k)),'b')
    legend('Median fitness','Minimum fitness')
    xlabel('Generation number')
    ylabel('Fitness')
    drawnow
    oldpop=pop;
    for i=1:f.popsize
        pop(i,:)=oldpop(isort(i),:);
    end
    
    %% Store population
    populations(:,:,k)=pop;
    %% Test convergence
    if k==f.maxgen
        disp(['Failed to converge after ' num2str(k) ' generations'])
        break
    end
    if k>f.stat && isequal((populations(1,:,k)),(populations(1,:,k-f.stat)))
        disp(['Stopping due to static population (' num2str(k) ' generations)'])
        break
    end
    if toc>f.maxtime
        disp(['Maximum time reached (' num2str(f.maxtime) ' seconds)'])
        break
    end
end
%% Evaluate class separation
X=X0;
T1=zeros(nsamp,1);
P=zeros(totvars,1);
for d=1:f.dim
    variables{d}=sort(pop(1,(1+f.nvars*(d-1)):f.nvars*d));
    [T_1,V_1,PPOUT]=projpursuit(X(:,variables{d}),1,f.opt); %Full PP with 100 guesses
    kurt(d)=PPOUT.K; %Store kurtosis
    scores(:,d)=T_1; %Store scores
    V2=zeros(totvars,1);
    V2(variables{d})=V_1; %Construct vectors
    vectors(:,d)=V2';
    if d<f.dim %Deflation
        t=X*V2;
        T1(:,d)=t;
        P(:,d)=X'*t/(t'*t);
        X=X0-(T1*P');
        if d==2 %Special deflation step to push into 8 quadrants
            t=(scores(:,1)).*(scores(:,2));
            T1(:,3)=t;
            P(:,3)=X'*t/(t'*t);
            X=X0-(T1*P');
        end
    end
end
%% Plotting
figure
if iscell(f.objclass) %Convert cell array to matrix if necessary
    f.objclass=cell2mat(f.objclass);
end
if size(f.classlist)>1 %Reduce objclass cell array to one dimension if necessary
    f.classlist=f.classlist(:,1);
end
if isempty(f.objclass)==false
    if isempty(f.classlist)
        for i=1:max(f.objclass) %Make generic group names if no classs names
            f.classlist{i}=['Group ' num2str(i)];
        end
    end
    Color=[255 57 33;215 25 232;53 33 255;29 156 207; ...
        12 178 85;122 43 12;0 0 0;29 84 74]./255;
    if f.dim==1
        for i=1:max(f.objclass) %Plot in 2d with colour
            plot(scores(f.objclass==i,1),scores(f.objclass==i,1),'.','Color',Color(i,:),'MarkerSize',5)
            hold on
        end
        legend(f.classlist,'Location','BestOutside')
    end
    if f.dim==2
        for i=1:max(f.objclass) %Plot in 2d with colour
            plot(scores(f.objclass==i,1),scores(f.objclass==i,2),'.','Color',Color(i,:),'MarkerSize',10)
            hold on
        end
        legend(f.classlist,'Location','BestOutside')
    elseif f.dim==3
        for i=1:max(f.objclass) %Plot in 3d with colour
            hold on
            plot3(scores(f.objclass==i,1),scores(f.objclass==i,2), ...
                scores(f.objclass==i,3),'.','Color',Color(i,:),'MarkerSize',10)
        end
        legend(f.classlist,'Location','BestOutside')
        hold off
    end
else
    if f.dim==2 %Plot all in black if no classes are given
        plot(scores(:,1),scores(:,2),'.k','MarkerSize',5)
    elseif f.dim==3
        plot3(scores(:,1),scores(:,2), ...
            scores(:,3),'.k','MarkerSize',5)
    end
end
toc
end
%% ------------Fitness based mating for sum/prod------------------%%
function [popchild]=mating6(nvars,pop,nchild,fitness,newpop,totvars,dim,ctoff,exponent)
popchild=zeros(nchild,nvars);
fitness(fitness<ctoff)=ctoff;
y=1./fitness.^exponent;
ranks=cumsum(y)/sum(y);
for i=1:2:nchild-1
    it=0;
    parents=zeros(2,nvars*dim);
    while (any(diff(sort(popchild(i,:)))==0)||any(diff(sort(popchild(i+1,:)))==0)) ... %no duplicate variables
            || sum(ismember(sort([newpop;popchild],2),sort(popchild(i,:),2),'rows'))>1 %no duplicate individuals
        while parents(1,:)==parents(2,:) %No duplicate parents
            parents(1,:)=pop(find(ranks>=rand(1),1),:); %Select parents based on cum. dist. function (ranks)
            parents(2,:)=pop(find(ranks>=rand(1),1),:);
        end
        for d=1:dim
            varsel=randperm(nvars*2);
            pool=parents(:,(1+(d-1)*(nvars)):d*nvars);
            popchild(i,(1+(d-1)*(nvars)):d*nvars)=pool(varsel(1:nvars));
            popchild(i+1,(1+(d-1)*(nvars)):d*nvars)=pool(varsel((nvars+1):(nvars*2)));
        end
        it=it+1;
        if it>30 %Generate random individual if it cannot find a new unique one after 30 iters.
            popchild(i,:)=randperm(totvars,nvars*dim);
            popchild(i+1,:)=randperm(totvars,nvars*dim);
        end
    end
end
end

%% -----------------Projection pursuit algorithm--------------------------------%%
function [ T,V,ppout ]=projpursuit(X,varargin)
%PROJPURSUIT  Projection Pursuit Analysis
%   T = PROJPURSUIT(X) performs projection pursuit analysis on the
%   matrix X, using default algorithmic parameters (see below) and
%   returns the scores in T.  The matrix X is mxn (objects x variables)
%   and T is mxp (objects x scores), where the default value of p is 2.
%
%   Projection pusuit (PP) is an exploratory data analysis technique that
%   seeks to optimize a projection index to find "interesting" projections
%   of objects in a lower dimensional space.  In this algorithm, kurtosis
%   (fourth statistical moment) is used as the projection index.
%
%   T = PROJPURSUIT(X,P) returns the first P projection pursuit scores.
%   Usually P is 2 or 3 for data visualization (default = 2).
%
%   T= PROJPURSUIT(X,P,GUESS) uses GUESS initial random starting points for
%   the optimization.  Larger values of GUESS decrease the likelihood of a
%   local optimum, but increase computation time.  The default value is
%   GUESS=100.
%
%   T = PROJPURSUIT(X,...,S1,S2,...) specifies algorithmic variation of
%   the PP analysis, where S1, S2, etc. are character strings as specified
%   with the options below.
%
%      Stepwise Unvariate ('Uni') or Multivariate ('Mul') Kurtosis
%      Ordinary ('Ord') or Recentered ('Rec') Kurtosis
%      Orthogonal Scores ('SO') or Orthogonal Loadings ('VO')
%      Minimization ('Min') or Maximization ('Max') of Kurtosis
%      Shifted ('Sh') or Standard ('St') Optimization Method
%
%   In each case, the default option is the first one.  These variations
%   are discussed in more detail below under the heading 'Algorithms'.
%
%   [T,V] = PROJPURSUIT(...) returns the P loading vectors in V (nxp).
%
%   [T,V,PPOUT] = PROJPURSUIT(...) returns additional outputs from the PP
%   analysis in the structured variable PPOUT. These vary with the
%   algorithm selected, as indicated below.
%        PPOUT.K:  Kurtosis value(s) for the optimum subspace. Can
%                  otherwise be found by searching for the max/min of
%                  PPOUT.kurtObj. For multivariate methods, this is a
%                  scalar; for univariate methods, it is a 1xP vector
%                  corresponding to the optimum value in each step.
%        PPOUT.kurtObj: Kurtosis values for different initial guesses.
%        PPOUT.convFlag: Convergence status for different initial guesses.
%        PPOUT.W:  If the scores are made orthogonal for univariate
%                  methods, W and P are intermediate matrices in the
%                  calculation of deflated matrices. The loadings are not
%                  orthogonal in this case and are given by V=W*inv(P'*W).
%                  If the projection vectors are set to be orthogonal, or
%                  multivariate algorithms are used, these are not
%                  calculated.
%        PPOUT.P:  See PPOUT.W.
%        PPOUT.Mu: The estimated row vector subtracted from the data
%                  set, X, for re-centered methods.
%
%   Algorithms:
%
%   Univariate vs. Multivariate
%      In the stepwise univariate PP algorithm, univariate kurtosis is
%      optimized as the projection vectors are extracted sequentially,
%      with deflation of the original matrix at each step. In the
%      multivariate algorithm, multivariate kurtosis is optimized as
%      all of the projection vectors are calculated simultaneously.
%      Univariate is best for small numbers of balanced clusters that can
%      be separated in a binary fashion and runs faster than the
%      multivariate algorithm.
%
%   Minimization vs Maximization
%      Minimization of kurtosis is most often used to identify clusters.
%      Maximization may be useful in identifying outliers. Maximization
%      is not an option for recentered algorithms.
%
%   Orthogonal Scores vs. Orthogonal Loadings
%      This option is only applicable to stepwise univariate algorithms
%      for P>1 and relates to the deflation of the data matrix in the
%      stepwise procedure. Orthogonal scores are generally preferred,
%      since these avoid correlated scores in multiple dimensions.
%      However, the projection vectors (loadings) will not be orthogonal
%      in this case.  For multivariate methods, the loadings are always
%      orthogonal.
%
%   Ordinary vs. Recentered Algorithms
%      For data sets that are unbalanced (unequal number of members in each
%      class, the recentered algorithms may provide better results than
%      ordinary PP.
%
%   Shifted vs. Standard Algorithms
%      This refers to the mathematics of the quasi-power method. The
%      shifted algorithm should be more stable, but the option for the
%      standard algorithm has been retained. The choice is not available
%      for recentered algorithms, and the shifted algorithm may still be
%      implemented if solutions become unstable.

%%
%                             Version 1.0
%
% Original algorithms written by Siyuan Hou.
% Additional modifications made by Peter Wentzell and Chelsi Wicks.
%

%% Set Default Parameters
MaxMin='Min';
StSh='Sh';
VSorth='SO';
Meth='Uni';
CenMeth='Ord';
p=2;
guess=100;
ppout.W=[];
ppout.P=[];
ppout.Mu=[];

%% Check for valid inputs and parse as required

if ~exist('X','var')
    error('PP:DefineVar:X','Provide data matrix X')
elseif ~isa(X,'double')
    error('PP:InvalVar:X','Invalid data matrix X')
end

% Extract numeric variables if present
opt_start=1;       % Marks beginning of algorithmic options in varargin
if nargin>1
    if isa(varargin{1},'double')   % Second argument is p?
        p=round(varargin{1});
        opt_start=2;
        if nargin>2
            if isa(varargin{2},'double')  % No. of guesses given?
                guess=round(varargin{2});
                opt_start=3;
            end
        end
    end
end

% Check numeric variables
[m,n]=size(X);         % Check numeric variables
if numel(p)~=1 || p<1  % Check if p is valid
    error('PP:InvalVar:p','Invalid value for subspace dimension.')
elseif numel(guess)~=1 || guess<1    % Check no. of guesses
    error('PP:InvalVar:guess','Invalid value for number of guesses.')
elseif m<(p+1) || n<(p+1)   % Check X
    error('PP:InvalVar:X','Insufficient size of data matrix.')
end

% Extract string variables if present
Allowd_opts='unimulordrecsovominmaxshst';
OptStrg='';                       % String to concatenate all options
for i=opt_start:size(varargin,2)
    if ischar(varargin{i})
        temp=lower(varargin{i});
        if isempty(strfind(Allowd_opts,temp))
            error('PP:InvalVar:OptStrg','Invalid option syntax.')
        end
        OptStrg=strcat(OptStrg,temp); %creates string of all character options
    else
        error('PP:InvalVar:OptStrg','Invalid option syntax.')
    end
end

% Set options for algorithm

if strfind(OptStrg,'max')
    if strfind(OptStrg,'min')
        error('PP:InvMode:MaxMin','Choose either to minimize or maximize.')
    elseif strfind(OptStrg,'rec')
        error('PP:InvMode:MaxMin','Maximization not available for recentered PP.')
    else
        MaxMin='Max';
    end
end

if strfind(OptStrg,'st')
    if strfind(OptStrg,'sh')
        error('PP:InvMode:StSh','Choose either the standard or shifted method')
    else
        StSh='St';
    end
end

if strfind(OptStrg,'vo')
    if strfind(OptStrg,'so')
        error('PP:InvMode:VSorth','Choose for either the scores or the projection vectors to be orthogonal')
    else
        VSorth='VO';
    end
end

if strfind(OptStrg,'mul')
    if strfind(OptStrg,'uni')
        error('PP:InvMode:UniMul','Choose either univariate or multivariate method')
    else
        Meth='Mul';
    end
end

if strfind(OptStrg,'rec')
    if strfind(OptStrg,'ord')
        error('PP:InvMode:OrdRec','Choose either the ordinary or recentred method')
    else
        CenMeth='Rec';
    end
end

%% Carry out PP using appropriate algorithm

if strcmp(Meth,'Mul')
    if strcmp(CenMeth,'Rec')
        %disp('Performing recentered multivariate PP')  % Diagnostic
        [T,V,R,K,Vall,kurtObj,convFlag]=rcmulkurtpp(X,p,guess);
        ppout.K=K;
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
        ppout.Mu=R;
    else
        %disp(['Performing ordinary multivariate PP(' StSh ')'])  % Diagnostic
        [T,V,Vall,kurtObj,convFlag]=mulkurtpp(X,p,guess,MaxMin,StSh);
        ppout.K=min(kurtObj);
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
    end
else
    if strcmp(CenMeth,'Rec')
        %disp(['Performing recentered univariate PP(' VSorth ')'])  % Diagnostic
        [T,V,R,W,P,kurtObj,convFlag]=rckurtpp(X,p,guess,VSorth);
        ppout.K=min(kurtObj);
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
        ppout.W=W;
        ppout.P=P;
        ppout.Mu=R;
    else
        %disp(['Performing ordinary univariate PP(' StSh ',' VSorth ')'])  % Diagnostic
        [T,V,W,P,kurtObj,convFlag]=okurtpp(X,p,guess,MaxMin,StSh,VSorth);
        ppout.K=min(kurtObj);
        ppout.kurtObj=kurtObj;
        ppout.convFlag=convFlag;
        ppout.W=W;
        ppout.P=P;
    end
end
end

%% Original Univariate Kurtosis Projection Pursuit Algorithm
function [T,V,W,P,kurtObj,convFlag]=okurtpp(X,p,guess,MaxMin,StSh,VSorth)
%% Quasi-power methods to optimize univariate kurtosis
%
%%
% Input:
%       X:       The data matrix. Rows denote samples, and columns denote variables.
%       p:       The number of projection vectors to be extracted.
%       guess:   The number of initial guesses for optimization,e.g. 100.
%                The more dimensions, the better to have more initial guesses.
%       MaxMin:  A string indicating to search for maxima or minima of kurtosis.
%                The available choices are "Max" and "Min".
%                   "Max": To search for maxima of kurtosis
%                   "Min": To search for minima of kurtosis
%                Projections revealing outliers tend to have a maximum
%                kurtosis, while projections revealing clusters tend to
%                have a minimum kurtosis. Maximization seems more important
%                in ICA to look for independent source signals, while
%                minimization appears useful in PP to looks for clusters.
%       StSh:    A string indicating if the standard or the shifted algorithm
%                is used. The available choices are "St" and "Sh".
%                   "St": To use the standard quasi-power method.
%                   "Sh": To use the shifted quasi-power method.
%       VSorth:  A string indicating whether the scores or projection
%                vectors are orthogonal. The available choices are
%                   "VO": The projection vectors are orthogonal, but
%                         scores are not, in general.
%                   "SO": The scores are orthogonal, but the projection
%                         vectors are not, in general.
%                If not specified (empty), the scores are made orthogonal.
% Output:
%       T:        Scores.
%       V:        Projection vectors.
%       W & P:    If the scores are made orthogonal, they appear in the
%                 deflation steps. They can be used to calculate the final
%                 projection vectors with respect to the original matrix.
%                 If the projection vectors are set to be orthogonal, they
%                 are not needed.
%       kurtObj:  Kurtosis values for different initial guesses.
%       convFlag: Convergence status for different initial guesses.

%% Note:
%
% The scores orthogonality is based on mean-centered data. If the data
% are not mean-centered, the mean scores are added to the final scores and
% therefore the final scores may not be not orthogonal.
%
% For minimization of kurtosis, the standard method (st) may not be stable
% when the number of samples is only slightly larger than the number of
% variables. Thus, the shifted method (sh) is recommended.

% Author:
% S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2012.
%
% Version, Nov. 2012. This is the updated version. The original version was
% reported in the literature: S. Hou, and P. D. Wentzell, Fast and Simple
% Methods for the Optimization of Kurtosis Used % as a Projection Pursuit
% Index, Analytica Chimica Acta, 704 (2011) 1-15.
%%
if exist('VSorth','var')
    if (strcmpi(VSorth,'VO')||strcmpi(VSorth,'SO'))
        % Pass
    else
        error('Please correctly choose the orthogonality of scores or projection vectors.')
    end
else
    VSorth='SO';
end
%
if strcmpi(StSh,'St') || strcmpi(StSh,'Sh')
    StSh0=StSh;
else
    error('Please correctly choose "St" or "Sh" method.')
end

%%  Mean center the data and reduce the dimensionality of the data
% if the number of variables is larger than the number of samples.
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig;
rk=rank(X);
if p>rk
    p=rk;
    display('The component number larger than the data rank is ignored.');
end

[Uorig,Sorig,Worig]=svd(X,'econ');
X=Uorig*Sorig;
X=X(:,1:rk);
Worig=Worig(:,1:rk);
X0=X;
%% Initial settings
[r,c]=size(X);
maxcount=10000;
convFlag=cell(guess,p);
kurtObj=zeros(guess,p);
T=zeros(r,p);
W=zeros(c,p);
P=zeros(c,p);
%%
for j=1:p
    cc=c+1-j;
    convlimit=(1e-10)*cc;         % Set convergence limit
    wall=zeros(cc,guess);
    [U,S,Vj]=svd(X,'econ');
    Vj=Vj(:,1:cc);                % This reduces the dimensionality of the data
    X=X*Vj;                       % when deflation is performed.
    if strcmpi(MaxMin,'Max')      % Option to search for maxima.
        invMat2=1./diag(X'*X);    % Note X'*X is diagonal due to SVD previously
    elseif strcmpi(MaxMin,'Min')  % Option to search for minima.
        Mat2=diag(X'*X);
        VM=zeros(cc*cc,r);        % This is used to calculate "Mat1a" later
        for i=1:r
            tem=X(i,:)'*X(i,:);
            VM(:,i)=reshape(tem,cc*cc,1);
        end
    else
        error('Please correctly choose to maximize or minimize the kurtosis.')
    end
    %% Loop for different initial guesses of w
    for k=1:guess
        w=randn(cc,1);   % Random initial guess of w for real numbers
        w=w/norm(w);
        oldw1=w;
        oldw2=oldw1;
        StSh=StSh0;
        count=0;
        while 1
            count=count+1;
            x=X*w;
            %% Maximum or minimum search
            if strcmpi(MaxMin,'Max')         % Option to search for maxima.
                w=invMat2.*(X'*(x.*x.*x));
            elseif strcmpi(MaxMin,'Min')     % Option to search for minima.
                Mat1=sum(VM*(x.*x),2);
                Mat1=reshape(Mat1,cc,cc);
                w=Mat1\(Mat2.*w);
            end
            %% Test convergence
            w=w/norm(w);
            L1=(w'*oldw1)^2;
            if (1-L1) < convlimit
                convFlag(k,j)={'Converged'};
                break   % Exit the "while ... end" loop if converging
            elseif count>maxcount
                convFlag(k,j)={'Not converged'};
                break   % Exit if reaching the maximum iteration number
            end
            %% Continue the interation if "break" criterion is not reached
            if strcmpi(StSh,'Sh')            % Shifted method
                w=w+0.5*oldw1;
                w=w/norm(w);
            elseif strcmpi(MaxMin,'Min')     % "St" method & minimization
                L2=(w'*oldw2)^2;             % If "St" method is not stable,
                if L2>L1 && L2>0.99          % change to shifted method
                    StSh='Sh';
                    display('Warning: "St" method is not stable. Change to shifted method.');
                end
                oldw2=oldw1;
            end                 % "St" method & maximization: do nothing
            oldw1=w;
        end
        %% Save the projection vectors for different initial guesses
        wall(:,k)=w;
    end
    %% Find the best solution from different initial guesses
    kurtObj(:,j)=kurtosis(X*wall,1,1);
    if strcmpi(MaxMin,'Max')        % Find the best projection vector for maximum search.
        [tem,ind]=max(kurtObj(:,j));
    elseif strcmpi(MaxMin,'Min')    % Find the best projection vector for minimum search.
        [tem,ind]=min(kurtObj(:,j));
    end
    Wj=wall(:,ind);                 % Take the best projection vector as the solution.
    
    %% Deflation of matrix
    if strcmpi(VSorth,'VO')       % This deflation method makes the
        t=X*Wj;                   % projection vectors orthogonal.
        T(:,j)=t;
        W(:,j)=Vj*Wj;
        X=X0-X0*W*W';
    elseif strcmpi(VSorth,'SO') % This deflation method makes the scores orthogonal.
        t=X*Wj;       % This follows the deflation method used in the non-linear partial
        T(:,j)=t;     % least squares (NIPALS), which is well-known in chemometrics.
        W(:,j)=Vj*Wj;
        Pj=X'*t/(t'*t);
        P(:,j)=Vj*Pj;
        X=X0-T*P';        % This uses the Gram-Schmidt process for complex-valued vectors
    end
end
%% Transform back into original space
W=Worig*W;         % The projection vector(s) are tranformed into original space.
if strcmpi(VSorth,'VO')
    V=W;
    W=[];
    P=[];
    T=T+Morig*V;    % Adjust the scores. Mean scores are added.
else
    P=Worig*P;      % Vectors in P are tranformed into original space.
    V=W*inv(P'*W);  % Calculate the projection vectors by V=W*inv(P'*W)
    T=T+Morig*V;    % Adjust the scores. Mean scores are added.
    tem=sqrt(sum(abs(V).^2));
    V=V./(ones(size(V,1),1)*tem); % Make the projection vectors be unit length
    T=T./(ones(size(T,1),1)*tem); % Adjust T with respect to V
    P=P.*(ones(size(P,1),1)*tem); % Adjust P with respect to V
end
end
%% =================== End of the function =======================
%%

%% Original Multivariate Kurtosis Projection Pursuit Algorithm
function [T,V,Vall,kurtObj,convFlag]=mulkurtpp(X,p,guess,MaxMin,StSh)
%
% Quasi-power method to optimize multivariate kurtosis.
%%
% Input:
%       X:      The data matrix.
%       p:      The dimension of the plane or heperplane (Normally, 2 or 3).
%       guess:  The number of initial guesses for optimization.
%               The more dimension, the better to have more initial guesses.
%       MaxMin: A string indicating to search for maxima or minima of kurtosis.
%               The available choices are "Max" and "Min".
%                   "Max": To search for maxima of kurtosis
%                   "Min": To search for minima of kurtosis
%               Projections revealing outliers tend to have a maximum
%               kurtosis, while projections revealing clusters tend to
%               have a minimum kurtosis.
%       StSh:   A string indicating if the standard or the shifted algorithm
%               is used. The available choices are "St" and "Sh".
%                   "St": To use the standard quasi-power method.
%                   "Sh": To use the shifted quasi-power method.
% Output:
%       T:        Scores.
%       V:        Projection vectors.
%       Vall:     All the projection vectors found based on different initial guesses. The
%                 best projection vectors are chosen as the solutions and put in V
%       kurtObj:  Kurtosis values for different projection vectors.
%       convFlag: Convergence status for the initial guesses..
%%
%  Mean center the data and reduce the dimensionality of the data if the number
%  of variables is larger than the number of samples.
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig;
rk=rank(X);
[Uorig,Sorig,Vorig]=svd(X,'econ');
X=Uorig*Sorig;
X=X(:,1:rk);
Vorig=Vorig(:,1:rk);
[r,c]=size(X);
%%
% Initial settings
maxcount=10000;
convlimit=1e-10;
Vall=cell(1,guess);
kurtObj=zeros(1,guess);
convFlag=cell(1,guess);
%%
for k=1:guess
    V=randn(c,p); % Random initial guess of V
    V=orth(V);
    oldV=V;
    count=0;
    while 1
        count=count+1;
        A=V'*X'*X*V;
        Ainv=inv(A);
        %%
        %         kurt=0;
        %         Mat=zeros(c,c);
        %         for i=1:r
        %             scal=(X(i,:)*V*Ainv*V'*X(i,:)');
        %             kurt=kurt+scal^2;
        %             Mat=Mat+scal*X(i,:)'*X(i,:);
        %         end
        scal=sqrtm(Ainv)*V'*X';
        scal=sqrt(sum(scal.^2,1));
        Mat=((ones(c,1)*scal).*X');
        Mat=Mat*Mat';
        % The four lines replace the above loop to increase the speed.
        
        %%
        if strcmpi(MaxMin,'Max')        % Option to search for maxima.
            M=inv(X'*X)*Mat;
            if strcmpi(StSh,'St')
                V=M*V;
            elseif strcmpi(StSh,'sh')
                V=(M+eye(c)*trace(M)/c)*V;
            else
                error('Please correctly choose to standard or shifted method.')
            end
        elseif strcmpi(MaxMin,'Min')    % Option to search for minima.
            M=inv(Mat)*(X'*X);
            if strcmpi(StSh,'St')
                V=M*V;
            elseif strcmpi(StSh,'sh')
                V=(M+eye(c)*trace(M)/c)*V;
            else
                error('Please correctly choose to standard or shifted method.')
            end
        else
            error('Please correctly choose to maximize or minimize the kurtosis.')
        end
        %%
        [V,TemS,TemV]=svd(V,'econ');        % Apply SVD to find an orthonormal basis.
        if sum((oldV-V).^2)/(c*p)<convlimit % Test convergence.
            convFlag(1,k)={'Converged'};
            break
        elseif count>maxcount
            convFlag(1,k)={'Not converged'};
            break
        end
        oldV=V;
    end
    kurtObj(1,k)=r * sum( (sum( (sqrtm(Ainv)*V'*X').^2, 1 ) ).^2 ); % Calculate kurtosis.
    %%
    [U,S,V]=svd(X*V*V','econ');
    Vall{1,k}=Vorig*V(:,1:p);
end
%%
if strcmpi(MaxMin,'Max')        % Find the best projection vector for maximum search.
    [tem,ind]=max(kurtObj(1,:));
elseif strcmpi(MaxMin,'Min')    % Find the best projection vector for minimum search.
    [tem,ind]=min(kurtObj(1,:));
end

V=Vall{1,ind};          % Store the projection vectors
T=X*Vorig'*V+Morig*V;   % Calculate the scores.
end
%% =================== End of the function =======================
%%

%% Recentered Univariate Kurtosis Projection Pursuit Algorithm
function [T,V,R,W,P,kurtObj,convFlag]=rckurtpp(X,p,guess,VSorth)
%
% Algorithms for minimization of recentered kurtosis. recentered kurtosis
% is proposed as a projection pursuit index in this work, aiming to deal with
% unbalanced clusters.
%
%%
% Input:
%       X:        The data matrix.
%       p:        The number of projection vectors to be extracted.
%       guess:    The number of initial guesses for optimization.
%                 The more dimensions, the better to have more initial guesses.
%       VSorth:   A string indicating whether the scores or projection
%                 vectors are orthogonal. The available choices are
%                   "VO": The projection vectors are orthogonal, but
%                         scores are not, in general.
%                   "SO": The scores are orthogonal, but the projection
%                         vectors are not, in general.
%                If not specified (empty), the scores are made orthogonal.
% Output:
%       T:        Scores.
%       V:        Projection vectors.
%       R:       The estimated row vector subtracted from the data set X.
%       W & P:    If users choose scores are orthogonal, they appear in the
%                 deflation steps. They can be used to calculate the final
%                 projection vectors with respect to the original matrix X.
%                 If the projection vectors are set to be orthogonal, they
%                 are not needed.
%       kurtObj:  Kurtosis values for different initial guesses.
%       convFlag: Convergence status for different initial guesses.

%% Note:
% Users have the option to make the projection vectors or scores orthogonal.
% The scores orthogonality is based on mean-centered data. If the data
% are not mean-centered, the mean scores are added to the final scores and
% therefore the final scores may not be not orthogonal.
%% Author:
% S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2012.
%
% This algorithm is based on the Quasi-Power methods. The Quasi-power
% methods were reported in the literature: S. Hou, and P. D. Wentzell,
% Fast and Simple Methods for the Optimization of Kurtosis Used as a
% Projection Pursuit Index, Analytica Chimica Acta, 704 (2011) 1-15.
%
%%
if exist('VSorth','var')
    if (strcmpi(VSorth,'VO')||strcmpi(VSorth,'SO'))
        % Pass
    else
        error('Please correctly choose the orthogonality of scores or projection vectors.')
    end
else
    VSorth='SO';
end
%
%%  Mean center the data and reduce the dimensionality of the data
% if the number of variables is larger than the number of samples.
Morig=mean(X);
X=X-ones(size(X,1),1)*Morig;
rk=rank(X);
if p>rk
    p=rk;
    display('The component number larger than the data rank is ignored.');
end
%
[Uorig,Sorig,Worig]=svd(X,'econ');
X=Uorig*Sorig;
X=X(:,1:rk);
Worig=Worig(:,1:rk);
X0=X;
%% Initial settings
[r,c]=size(X);
maxcount=10000;
convFlag=cell(guess,p);
kurtObj=zeros(guess,p);
T=zeros(r,p);
W=zeros(c,p);
P=zeros(c,p);
ALPH=zeros(1,p);
%%
for j=1:p
    cc=c+1-j;
    convlimit=(1e-10)*cc;         % Set convergence limit
    wall=zeros(cc,guess);
    alphall=zeros(1,guess);
    [U,S,Vj]=svd(X,'econ');
    Vj=Vj(:,1:cc);                % This reduces the dimensionality of the data
    X=X*Vj;                       % when deflation is performed.
    for k=1:guess
        w=randn(cc,1);   % Random initial guess of w for real numbers
        w=w/norm(w);
        alph=mean(X*w);
        oldw1=w;
        oldw2=oldw1;
        count=0;
        while 1
            count=count+1;
            x=X*w;
            xalph=(x-alph);
            alph=alph + sum(xalph.^3) / (3*sum(xalph.^2)); % Update alpha (alph) value
            mu=alph*w';                 % Updata mu, given w and alpha (alph)
            tem=(x-alph).^2;
            dalph_dv=(X'*tem)/sum(tem); % Calculate dalpha/dv
            tem1=X'-dalph_dv*ones(1,r);
            tem2=X-ones(r,1)*mu;
            Mat1=((ones(cc,1)*tem').*(tem1))*(tem2);
            Mat2=tem1*tem2;
            w=Mat1\(Mat2*w);            % updata w
            %% Test convergence
            w=w/norm(w);
            L1=(w'*oldw1)^2;
            if (1-L1) < convlimit
                convFlag(k,j)={'Converged'};
                break   % Exit the "while ... end" loop if converging
            elseif count>maxcount
                convFlag(k,j)={'Not converged'};
                break   % Exit if reaching the maximum iteration number
            end
            %% Continue the interation if "break" criterion is not reached
            L2=(w'*oldw2)^2;
            if L2>L1 && L2>0.95
                w=w+(rand/5+0.8)*oldw1;
                w=w/norm(w);
            end
            oldw2=oldw1;
            oldw1=w;
        end
        %% Save the projection vectors for different initial guesses
        wall(:,k)=w;
        alphall(1,k)=alph;
    end
    %% Take the best projection vector as the solution
    kurtObj(:,j)=( r*sum((X*wall-ones(r,1)*alphall).^4) ./ ( (sum((X*wall-ones(r,1)*alphall).^2)) .^2) )';
    [tem,ind]=min(kurtObj(:,j));
    Wj=wall(:,ind);               % Take the best projection vector as the solution.
    for i=1:cc
        if Wj(i)~=0;
            signum=sign(Wj(i));      % Change the sign of w to make it unique
            break
        end
    end
    Wj=Wj*signum;
    ALPH(1,j)=alphall(1,ind)*signum;
    %% Deflation of matrix
    if strcmpi(VSorth,'VO')       % This deflation method makes the
        t=X*Wj;                   % projection vectors orthogonal.
        T(:,j)=t;
        W(:,j)=Vj*Wj;
        X=X0-X0*W*W';
    elseif strcmpi(VSorth,'SO') % This deflation method makes the scores orthogonal.
        t=X*Wj;       % This follows the deflation method used in the non-linear partial
        T(:,j)=t;     % least squares (NIPALS), which is well-known in chemometrics.
        W(:,j)=Vj*Wj;
        Pj=X'*t/(t'*t);
        P(:,j)=Vj*Pj;
        X=X0-T*P';
    end
end
%% Transform back into original space
W=Worig*W;         % The projection vector(s) are tranformed into original space.
if strcmpi(VSorth,'VO')
    V=W;
    W=[];
    P=[];
    T=T+ones(r,1)*Morig*V;        % Adjust the scores. Mean scores are added.
    R=ALPH*V'+Morig;
else
    P=Worig*P;      % Vectors in P are tranformed into original space.
    V=W*inv(P'*W);  % Calculate the projection vectors by V=W*inv(P'*W)
    T=T+ones(r,1)*Morig*V;        % Adjust the scores. Mean scores are added.
    R=ALPH*(P'*W)*W'+Morig;
    tem=sqrt(sum(abs(V).^2));
    V=V./(ones(size(V,1),1)*tem); % Make the projection vectors be unit length
    T=T./(ones(size(T,1),1)*tem); % Adjust T with respect to V
    P=P.*(ones(size(P,1),1)*tem); % Adjust P with respect to V
end
end
%% =========================== End of the function ============================
%%

%% Recentered Multivariate Kurtosis Projection Pursuit Algorithm
function [T,V,R,K,Vall,kurtObj,convFlag]=rcmulkurtpp(X,p,guess)
%
% Algorithms for minimization of re-centered multivariate kurtosis that is
% used as a project pursuit index. This algorithm aims to deal with
% unbalanced clusters (multivariate version). The effect of dimension is
% taken into account by introducing a dimension term in the constraint.
%%
% Input:
%       X:      The data matrix. X cannot be singular.
%       p:      The dimensionality of the plane or heperplane (Normally, 2 or 3).
%       guess:  The number of initial guesses for optimization.
% Output:
%       T:      Scores of the chosen subspace (with the lowest multivariate
%               kurtosis value).
%       V:      Projection vectors for the chosen subspace.
%       R:      The estimated row vector subtracted from the data set X.
%       K:      Multivariate kurtosis value for the chosen subspace.
%       Vall:   All the projection vectors found based on different initial guesses. The
%               best projection vectors are chosen as the solutions and put in V.
%       kurtObj:   Kurtosis values for the projection vectors of different initial guesses.
%       convFlag: Convergence status for the different initial guesses.
%
%%
% This algorithm extends the Quasi-Power methods reported in two papers:
% (1) S. Hou, and P. D. Wentzell, Fast and Simple Methods for the Optimization
%     of Kurtosis Used as a Projection Pursuit Index, Analytica Chimica Acta,
%     704 (2011) 1-15. (featured article)
% (2) S. Hou, and P. D. Wentzell,Re-centered Kurtosis as a Projection Pursuit
%     Index for Multivariate Data Analysis, Journal of Chemometrics, 28
%     (2014) 370-384.   (Special issue article)
%
% Author:
% S. Hou, University of Prince Edward Island, Charlottetown, PEI, Canada, 2014.
%
%% Mean-center the data
[n,m]=size(X);
Morig=mean(X);
X=X-ones(n,1)*Morig;

%% Initial settings
maxcount=10000;
convlimit=1e-10;
Vall=cell(1,guess);
rall=cell(1,guess);
kurtObj=zeros(1,guess);
convFlag=cell(1,guess);

%% Loop
for i=1:guess
    count=0;
    V=randn(m,p);       % Random initial guess of V
    V=orthbasis(V);
    oldV1=V;
    R=mean(X)';
    while 1
        count=count+1;
        
        %% Update r
        Y=(X-ones(n,1)*R'/p)*V;     % Note p is in the denominator
        invPsi=inv(Y'*Y);
        gj=diag(Y*invPsi*Y');
        Yj=Y*invPsi*(sum(Y))';
        J=(2* Y'* ((Yj*ones(1,p)).*Y) * invPsi - eye(p)*(sum(gj)+2))/p;  % Jacobian matrix
        f=sum(Y'.*(ones(p,1)*gj'),2);
        R=R-V*(J\f);                % Newton' method
        
        %% Update V
        % Calculate b1 and b2
        XX=X-ones(n,1)*R';          % Note p is not in the denominator
        Z=XX*V;
        S=Z'*Z;
        invS=inv(S);
        ai=diag(Z*invS*Z');
        Z_ai=(ai*ones(1,p)).*Z;
        Si_ai=Z'*Z_ai;
        
        b1=-J'\(invS*Si_ai*invS* (sum(Z))');
        b2=-J'\(invS*(sum(Z_ai))');
        
        % Calculate the 8 matrices
        Yj_b1_Yj=(Y*b1*ones(1,p)).*Y;
        Yj_b2_Yj=(Y*b2*ones(1,p)).*Y;
        Xj_gj=sum((gj*ones(1,m)).*X);
        
        M1=X'*Z*invS*Si_ai;
        M2=-Xj_gj'*b1'*S;
        M3=2*X'*Y*(invPsi*Y'*Yj_b1_Yj*invPsi*S);     % Parentheses added to speed up
        M4=-2*X'*Yj_b1_Yj*invPsi*S;
        
        M5=(X'.*(ones(m,1)*ai'))*XX;                 % Full rank
        M6=-Xj_gj'*b2'*Z'*XX;                        % Not full rank
        M7=2*X'*Y*(invPsi*Y'*Yj_b2_Yj*invPsi*Z'*XX); % Parentheses added to speed up
        M8=-2*X'*Yj_b2_Yj*invPsi*Z'*XX;
        
        % Calculate new V
        V=(M5+M6+M7+M8)\(M1+M2+M3+M4);
        V=orthbasis(V);
        
        % Test convergence
        L=abs(V)-abs(oldV1);
        L=trace(L'*L);
        if L<convlimit*p
            convFlag(1,i)={'Converged'};
            break
        elseif count>maxcount
            convFlag(1,i)={'Not converged'};
            break
        end
        oldV1=V;
    end
    
    % Save the subspaces for different initial guesses. Note the basis of the
    % subspace has been changed in accordance with PCA (mean-centered) criterion.
    kurtObj(1,i)=n*sum(diag(Z*inv(Z'*Z)*Z').^2);
    [Utem,Stem,Vtem]=svd(X*V,'econ'); % X has been mean-centered.
    Vtem=V*Vtem;
    Vall(1,i)={Vtem};
    rall(1,i)={(R'*Vtem*Vtem')}; % r is saved as a row vector now.
end

%% Take the best projection vector as the solution
[tem,ind]=min(kurtObj);
V=Vall{ind};
R=rall{ind};
T=X*V;
K=kurtObj(ind);

%% Add mean value
T=T+ones(n,1)*Morig*V; % Adjust the scores (The scores of mean vector are added).
R=R+Morig;             % Adjust r (mean vector is added).
end
%% ============== End of the function =====================

function [V]=orthbasis(A)
% Calculate an orthonormal basis for matix A using Gram-Schimdt process
% Reference: David Poole, Linear Algebra - A Modern Introduction,
% Brooks/Cole, 2003. pp.376.
%
% Input:
%   A: a matrix
% Output:
%   V: an orthonormal matrix

%%
c=size(A,2);
V(:,1)=A(:,1)/norm(A(:,1));
for i=2:c
    tem=A(:,i)-V*V'*A(:,i);
    V(:,i)=tem/norm(tem);
end
end
