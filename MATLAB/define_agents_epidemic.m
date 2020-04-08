function [agents,agent_states,no_agents,L_o_A]=define_agents_epidemic(data_locations,T_sim)

ref_indices_locations=data_locations.ref_indices_locations;
no_types_locations=data_locations.no_types_locations;


agents=[];
agent_states=[];
no_agents=[];


% reference indexek, ahol a location-ok listajaban az adott tipusu 
%location-ok indexei kezdodnek (ez az ertek + 1 felel meg az elso ilyen tipusu location-nek)
ref_inex_residences=ref_indices_locations.ref_inex_residences;
ref_inex_schools=ref_indices_locations.ref_inex_schools;
ref_inex_workplace_small=ref_indices_locations.ref_inex_workplace_small;
ref_inex_workplace_large=ref_indices_locations.ref_inex_workplace_large;
ref_inex_shops=ref_indices_locations.ref_inex_shops;
ref_inex_city_park=ref_indices_locations.ref_inex_city_park;
ref_inex_cinema=ref_indices_locations.ref_inex_cinema;
ref_inex_health_centre=ref_indices_locations.ref_inex_health_centre;
ref_inex_hospital=ref_indices_locations.ref_inex_hospital;


no_residence=no_types_locations.no_residence;
no_schools=no_types_locations.no_schools;
no_workplace_small=no_types_locations.no_workplace_small;
no_workplace_large=no_types_locations.no_workplace_large;
no_shops=no_types_locations.no_shops;
no_city_park=no_types_locations.no_city_park;
no_cinema=no_types_locations.no_cinema;
no_health_centre=no_types_locations.no_health_centre;
no_hospital=no_types_locations.no_hospital;


%===========

no_agents_type_1=35;

no_agents.no_agents_type_1=no_agents_type_1;

block_start_index=0; % az kovetkezo tipusu agent-ek indexelese honnan kezdodik az agent-ek listajaban
for i=1:no_agents_type_1

    agent_akt.age=randi([30 45]);
    agent_akt.flag_chronic=0;
    agent_akt.type=1; % see the interpretations in define_types_epidemic.m
    
    % type related parameters. Ez fontos itt kell megadni a tipushoz
    % tartozo parametereket: ezek igazabol location-ok indexei
    agent_akt.type_related_pars.home = ref_inex_residences+ i - no_residence*floor((i-1)/no_residence);% itt most az i-k agent otthona az i-k residence, ha mindegyikben van mar ujrakezdi
    agent_akt.type_related_pars.school = ref_inex_schools + i - no_schools*floor((i-1)/no_schools);  % , hasonloan
    agent_akt.type_related_pars.workplace = ref_inex_workplace_small + i - no_workplace_small*floor((i-1)/no_workplace_small); % ezek most kis munkahelyen dolgoznak
    agent_akt.type_related_pars.preferred_shop_1 = ref_inex_shops + i - no_shops*((i-1)/no_shops); 
    agent_akt.type_related_pars.preferred_shop_2 = ref_inex_shops + randi([1 no_shops]);
    agent_akt.type_related_pars.preferred_cinema = ref_inex_cinema +1; % 
    agent_akt.type_related_pars.preferred_city_park = ref_inex_city_park +1;
    agent_akt.type_related_pars.preferred_visit_site_1 = agent_akt.type_related_pars.home + 1; % thf mindenki 1-el inkrementalisan latogat
    agent_akt.type_related_pars.preferred_doctor = ref_inex_health_centre +1;
    agent_akt.type_related_pars.preferred_hospital = ref_inex_hospital +1;
    
    agent_akt.awareness_level=0;  % mondjuk h [0-5], 0 a baseline viselkedes amikor nincs jarvany
    
    %disp(['home: ' num2str(agent_akt.type_related_pars.home)])
    
    agents{i}=agent_akt;
    
    %============
    
   % agent_states_akt.location - a location-t kulon taroljuk egy
   % L_o_A: Location_of_Agents matrixban
   %T_sim
   L_o_A(block_start_index+i,:)=zeros(1,T_sim+1); % azert T_sim+1, mert vizsgaljk a t+1 indexet az utolso lepesben is
   L_o_A(block_start_index+i,1)=agent_akt.type_related_pars.home; % 
   
   
   %====================
   
   % kodok
   % S - S
   % I1 - 1
   % I2 - 2
   % I3 - 3
   % I4 - 4
   % I5 - 5
   % Recovered, immune - I
   % Recovered, reduced immunity - R
   % dead - D
   
   if mod(i,20)==0
       agent_states_akt.PP(1)='2'; 
   else
       agent_states_akt.PP(1)='S';
   end
   
   agent_states_akt.WB(1)='W';
   agent_states_akt.PPsc(1)=0; % PP state-counter: miota van ebben a PP state-ben
   agent_states_akt.flag_diagnosed(1)=0;
   
   agent_states{i}=agent_states_akt;
   
end
    
        
    
    
    
    
    
  