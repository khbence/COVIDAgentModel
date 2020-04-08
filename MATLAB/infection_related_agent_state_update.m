function agent_states_new=infection_related_agent_state_update(agent_index,agents,agent_states_akt,L_o_A,data_locations,t,agent_states)

global no_infections_at_locations

location_index_akt=L_o_A(agent_index,t);

locations=data_locations.locations;

location_akt=locations{location_index_akt};

area_akt=location_akt.area;

agent_akt=agents{agent_index};

age=agent_akt.age;
flag_chronic=agent_akt.flag_chronic;


%[t agent_index]
%agent_states_akt.PP

if agent_states_akt.PP(t)=='S';
    
   %kik vannakl meg ugyanazon helyen 
   
   agent_indices_present=find(L_o_A(:,t)==location_index_akt);
   
   % itt most csak nagyon egyszeruen megnezzuk a barmilyen mertekben
   % fertozottek suruseget - kesobb finomitani kell, akinek nincsennek
   % tunetei (I1 I2) az kevesbe, aki mar nincs olyan jol (I4, I5) az jobban
   % fertoz.
   indices_infected_agents_present=[];
   states_of_indices_infected_agents_present=[];
   
   for i=1:length(agent_indices_present)
       if agent_indices_present(i)~=agent_index
       
           state_of_other_agent=agent_states{agent_indices_present(i)};
           
           % tegyuk fel h inkubacio alatt nem fertoz
           if  state_of_other_agent.PP(t)=='2' | state_of_other_agent.PP(t)=='3' | state_of_other_agent.PP(t)=='4' | state_of_other_agent.PP(t)=='5'
               indices_infected_agents_present=[indices_infected_agents_present agent_indices_present(i) ]; 
               states_of_indices_infected_agents_present=[states_of_indices_infected_agents_present state_of_other_agent.PP(t)];
           end

       end
   end
    
   num_infected_agents_present=length(indices_infected_agents_present); % lehet pl kisebb/nagyobb sulyal figyelembe venni a kulonfele tipusu I-ket, es akkor ez megjelenik a density-ben
   
    density_of_infected=num_infected_agents_present/area_akt;
    
    if density_of_infected==0  % nehogy a nemlin fv numerikusan bekavarjon itt nekunk
        
        agent_states_akt.PP(t+1)='S';
        agent_states_akt.WB(t+1)='W';
        agent_states_akt.PPsc=agent_states_akt.PPsc+1;  
        agent_states_akt.flag_diagnosed=0;
        
    else
        
        prob_of_infection_akt=prob_infection(density_of_infected);
        if rand<prob_of_infection_akt   %<======================================== Ez is lassithat: elore kene random szamokat generalni, venni belole egyet, utana novelni egy globalis counter-t
            % =====INFECTION=========
            %========== display data for debugging (DDFD)
%             [agent_index, location_index_akt]
%             
%             agent_indices_present
%             indices_infected_agents_present
%             
%             L_o_A((agent_indices_present),t)
%             
%             [day_akt,hour_akt,minute_akt]=re_convert_time(t);
%             disp([num2str(day_akt) ':' num2str(hour_akt) ':' num2str(minute_akt)])
%             
%             pause
            %===============
            
            agent_states_akt.PP(t+1)='1';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;
            
            no_infections_at_locations(location_index_akt)=no_infections_at_locations(location_index_akt)+1;
            
        else
            
            agent_states_akt.PP(t+1)='S';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
            agent_states_akt.flag_diagnosed=0;
            
        end
    end
    
else
    agent_states_akt.PP(t+1)=agent_states_akt.PP(t);
    agent_states_akt.WB(t+1)=agent_states_akt.WB(t);
    agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
    agent_states_akt.flag_diagnosed=0;  % ezt majd kidolgozni
 
    
    
end

agent_states_new=agent_states_akt; 

end

function y=prob_infection(x)

global virulency

p=0.35-virulency;   % ezen parameterekkl lehet hangolni a virus fertozoseget (ha kirajzoljuk a fv-t, latni hogy hogyan hatnak ra)
 k=p/5;

y=1/(1+exp((p-x)/k));

end    

function [day,hour,minute]=re_convert_time(t)
      % [ora,perc -> 1-144]
      
      t=t-1;
      
      day=floor(t/144);
      time_in_actual_day=mod(t,144);
      hour=floor(time_in_actual_day/6);
      minute=mod(time_in_actual_day,6)*10;
      
end