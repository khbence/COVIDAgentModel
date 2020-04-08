function type_out=define_type_1_epid;

% type 1 - middle aged woring man with child

% related parameters of this type: These must be defined if an agent of this type is defined as 'type_related_pars'
%- these are used in the update process of the agent)
% these must be defined as location indices 
% home (must be a location index belonging to a residence type location)
% school (must be a location index belonging to a school type location)
% workplace (...)
% preferred_shop_1
% preferred_shop_2
% preferred_cinema (ebbol elvileg lehet tobb is)
% preferred_city_park (ebbol elvileg lehet tobb is)
% preferred_visit_site_1 ( residence type)
% preferred_doctor (ebbol elvileg lehet tobb is)
% preferred_hospital (igazabol nem preferred hanem amelyikhez tartozik, de mind1 -  ebbol elvileg lehet tobb is)

clc

type_out=[];

% type{j,k} - j-ik WB state, k=1 ha hetkoznap, k=2 ha hetvege


%==========================================================
% WB-state= W, hetkoznap: {1,1}:

index_akt=[1,1];

% 1. sor - otthon
% 2. sor - iskola
% 3. sor - munkahely
% 4. sor - uzlet 1
% 5. sor - uzlet 2
home=1;
school=2;
workplace=3;
shop_1=4;
shop_2=5;


M=zeros(5,144);


M=add_certain_location(M,home,[0,0,7,0]); % from 0:00 -to 7:00 certinly at home

M=add_transition(M,home,school,[7,10,7,30]); % may head for school 

M=add_certain_location(M,school,[7,40,8,0]); % for sure heads for school

M=add_certain_location(M,workplace,[8,10,15,0]);

M=add_transition(M,workplace,school,[15,10,15,40]); % may head for school 

M=add_certain_location(M,school,[15,50,15,50]);

M=add_equal_chances(M,[shop_1,shop_2,home],[16,00,18,00]);

M=add_certain_location(M,home,[18,10,23,50]);

%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    sum(M)
    error('prob sum not equal to one for some time index')
end

%==========================================================
% % WB-state= W, hetvege: {1,2}:

index_akt=[1,2];

% 1. sor - otthon
% 2. sor - uzlet 1
% 3. sor - uzlet 2
% 4. sor - city_park (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
% 5. sor - cinema (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
% 6 sor - latogatas
home=1;
shop_1=2;
shop_2=3;
city_park=4;
cinema=5;
visit=6;

M=zeros(6,144);

M=add_certain_location(M,home,[0,0,9,0]); % from 0:00 -to 7:00 certinly at home

M=add_equal_chances(M,[shop_1,shop_2,home],[9,10,11,30]);

M=add_certain_location(M,home,[11,40,13,30]);

M=add_equal_chances(M,[city_park,cinema,home,visit],[13,40,19,00]);

M=add_certain_location(M,home,[19,10,23,50]);

%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end



%==========================================================
% WB-state= N, hetkoznap: {2,1}:

index_akt=[2,1];

% 1. sor - otthon
% 2. sor - iskola
% 3. sor - munkahely
% 4. sor - uzlet 1
% 5. sor - uzlet 2
home=1;
school=2;
workplace=3;
shop_1=4;
shop_2=5;


M=zeros(5,144);


M=add_certain_location(M,home,[0,0,7,0]); % from 0:00 -to 7:00 certinly at home

M=add_transition(M,home,school,[7,10,7,30]); % may head for school 

M=add_certain_location(M,school,[7,40,8,0]); % for sure heads for school

M=add_weighted_chances(M,[workplace,home],[8,10,15,0],[0.75,0.25]);

M=add_transition(M,workplace,school,[15,10,15,40]); % may head for school 

M=add_certain_location(M,school,[15,50,15,50]);

M=add_equal_chances(M,[shop_1,shop_2,home],[16,00,18,00]);

M=add_certain_location(M,home,[18,10,23,50]);

%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    sum(M)
    error('prob sum not equal to one for some time index')
end

%==========================================================
% % WB-state= N, hetvege: {2,2}:

index_akt=[2,2];

% 1. sor - otthon
% 2. sor - uzlet 1
% 3. sor - uzlet 2
% 4. sor - city_park (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
% 5. sor - cinema (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
home=1;
shop_1=2;
shop_2=3;
city_park=4;
cinema=5;

M=zeros(5,144);

M=add_certain_location(M,home,[0,0,9,0]); % from 0:00 -to 7:00 certinly at home

M=add_weighted_chances(M,[shop_1,shop_2,home],[9,10,11,30],[0.3,0.3,0.4]);

M=add_certain_location(M,home,[11,40,13,30]);

M=add_weighted_chances(M,[city_park,cinema,home],[13,40,17,00],[0.2,0.2,0.6]);

M=add_certain_location(M,home,[17,10,23,50]);

%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end


%==========================================================
% WB-state= M, hetkoznap: {3,1}:
index_akt=[3,1];

% 1. sor - otthon
% 2. sor - iskola
% 3. sor - munkahely
% 4. sor - orvos
home=1;
school=2;
workplace=3;
doctor=4;


M=zeros(4,144);


M=add_certain_location(M,home,[0,0,7,30]); % from 0:00 -to 7:00 certinly at home

M=add_equal_chances(M,[home,school],[7,40,8,0]); % nem biztos hogy o viszi a gyereket

M=add_weighted_chances(M,[workplace,home,doctor],[8,10,15,0],[0.1,0.4,0.5]);

M=add_transition(M,workplace,school,[15,10,15,40]); % may head for school 

M=add_equal_chances(M,[home,school],[15,50,15,50]);

M=add_weighted_chances(M,[shop_1,shop_2,home],[16,00,18,00],[0.1,0.1,0.8]);

M=add_certain_location(M,home,[18,10,23,50]);

%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end


%==========================================================
% % WB-state= M, hetvege: {3,2}:

index_akt=[3,2];

% 1. sor - otthon
% 2. sor - doctor
home=1;
doctor=2;


M=zeros(2,144);

M=add_certain_location(M,home,[0,0,9,0]); % from 0:00 -to 7:00 certinly at home

M=add_weighted_chances(M,[home,doctor],[9,10,11,30],[0.6,0.4]);

M=add_certain_location(M,home,[11,40,23,50]);

%==========

%M

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end


%==========================================================
% % WB-state= S, hetkoznap: {4,1}:


index_akt=[4,1];

% 1. sor - korhaz
hospital=1;


M=zeros(1,144);

M=add_certain_location(M,hospital,[0,0,23,50]); % from 0:00 -to 7:00 certinly at home


%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end

%==========================================================
% % WB-state= S, hetvege: {4,2}:


index_akt=[4,2];

% 1. sor - korhaz
hospital=1;


M=zeros(1,144);

M=add_certain_location(M,hospital,[0,0,23,50]); % from 0:00 -to 7:00 certinly at home


%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end

%==========================================================
% % WB-state= D, hetkoznap: {5,1}:


index_akt=[5,1];

% 1. sor - korhaz
hospital=1;


M=zeros(1,144);

M=add_certain_location(M,hospital,[0,0,23,50]); % from 0:00 -to 7:00 certinly at home


%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end

%==========================================================
% % WB-state= S, hetvege: {5,2}:


index_akt=[5,2];

% 1. sor - korhaz
hospital=1;


M=zeros(1,144);

M=add_certain_location(M,hospital,[0,0,23,50]); % from 0:00 -to 7:00 certinly at home


%==========

flag_M_ok=check_consitency(M);
if flag_M_ok==1
    type_out{index_akt(1),index_akt(2)}=M;
else
    error('prob sum not equal to one for some time index')
end



end
%==============================================================================================
%==============================================================================================
%==============================================================================================
%==============================================================================================
% ezek a fuggvenyek az M matrix konnyebb feltoltesere valok:

function M_out=add_certain_location(M,rowindex,time_interval)

    M_out=M;
    % time interval: [8,30,14,40]: from 8:30 to 14:40
    t_1=convert_time(time_interval(1:2));
    t_2=convert_time(time_interval(3:4));
    
    M_out(rowindex,t_1:t_2)=ones(1,(t_2-t_1)+1);


end

function M_out=add_equal_chances(M,rowindices,time_interval)

    M_out=M;
    % time interval: [8,30,14,40]: from 8:30 to 14:40
    t_1=convert_time(time_interval(1:2));
    t_2=convert_time(time_interval(3:4));
    
    num_alternatives=length(rowindices);
    
    p_akt=1/num_alternatives;
    
    for i=1:num_alternatives
    
        M_out(rowindices(i),t_1:t_2)=p_akt*ones(1,(t_2-t_1)+1);
        
    end

end

function M_out=add_weighted_chances(M,rowindices,time_interval,weights)

    if sum(weights)~=1
        error('sum of weights must be 1')
    end

    M_out=M;
    % time interval: [8,30,14,40]: from 8:30 to 14:40
    t_1=convert_time(time_interval(1:2));
    t_2=convert_time(time_interval(3:4));
    
    num_alternatives=length(rowindices);
    
    for i=1:num_alternatives
    
        M_out(rowindices(i),t_1:t_2)=weights(i)*ones(1,(t_2-t_1)+1);
        
    end

end


function M_out=add_transition(M,rowindex_1,rowindex_2,time_interval)
    
    M_out=M;

    t_1=convert_time(time_interval(1:2));
    t_2=convert_time(time_interval(3:4));
    l_t=(t_2-t_1)+1;
    
    incr=1/(l_t+1);
    
    if l_t>1
    
        for i=1:l_t;
            M_out(rowindex_1,t_1+i-1)=1-incr*i;
            M_out(rowindex_2,t_1+i-1)=incr*i;
        end
            
    else
                
    end

end

% idokonverzio [ora,perc] formatumrol t \in [1-144]-ra

function t_out=convert_time(t_vect_in)
      % [ora,perc -> 1-144]
      
      t_out=[(t_vect_in(1)*60 + t_vect_in(2))/10]+1;
      
end

% kijon-e hogy 1 a valsegek osszege

function y=check_consitency(M);

y=1;

if size(M,1)>1
    sum_M=sum(M);
else
    sum_M=M;
end

for i=1:size(M,2)
    if abs(sum_M(i)-1)>10^-9 % numerikus dolgok miatt
        erroneous_column=i
        y=0;
    end
end
    


end


