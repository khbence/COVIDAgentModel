function agent_states_new=disease_progression_related_agent_state_update(agent_index,agents,agent_states_akt,L_o_A,data_locations,t,agent_states)


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

global probability_of_not_getting_sick_when_infected
 
global probability_of_I2_I3_transition

global probability_of_I3_I4_transition

global probability_of_I4_I5_transition

global morbidity_in_I5

global t_min_immunity

location_index_akt=L_o_A(agent_index,t);

locations=data_locations.locations;

location_akt=locations{location_index_akt};

area_akt=location_akt.area;

agent_akt=agents{agent_index};

age=agent_akt.age;
flag_chronic=agent_akt.flag_chronic;

% agent_index 
% agent_states_akt.PPsc
% agent_states_akt.PP
% pause

if agent_states_akt.PP(t)=='1';  % ha inkubacios periodusben van, egy ido utan atmegy I2-be  
        
    
        prob_trans_akt=prob_trans_from_latency(agent_states_akt.PPsc/144);% az a fv napokra vonatkozoan van megadva
        
%         disp('number of days in actual state , prob')
%         [agent_states_akt.PPsc/144 prob_trans_akt]
%         pause
        
        if rand<prob_trans_akt  %<============================================= ez is lassithat (randomgeneralas)
            agent_states_akt.PP(t+1)='2'; % transition from incubation 2 infected withoth symptoms
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0; % a flag diagnosed pl akkor update-elodhet majd 1-re ha orvosnal jar, vagy ha mashogy tesztelik
        else
            if floor(agent_states_akt.PPsc/144)>t_min_immunity % ezt napban kell merni!
                % ha mar regota van inkubacioban, bizonyos valseggel menjen at rezisztensbe
                if rand<probability_of_not_getting_sick_when_infected
                    agent_states_akt.PP(t+1)='I'; % immune
                    agent_states_akt.WB(t+1)='W';
                    agent_states_akt.PPsc=0; 
                    agent_states_akt.flag_diagnosed=0;
                else
                    agent_states_akt.PP(t+1)='1'; 
                    agent_states_akt.WB(t+1)='W';
                    agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
                    agent_states_akt.flag_diagnosed=0;
                end
            else
                agent_states_akt.PP(t+1)='1'; 
                agent_states_akt.WB(t+1)='W';
                agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
                agent_states_akt.flag_diagnosed=0;
            end
        end
elseif agent_states_akt.PP(t)=='2'
         
        % atmegy-e olyan allapotba ami tuneteket mutat (I3)
        
         if age>60
             offset=0.2;
         else
             offset=0; % lehet meg tobb eset is..
         end
         
         trans_prob=probability_of_I2_I3_transition+offset;
         
         if rand<trans_prob
             
            agent_states_akt.PP(t+1)='3';
            agent_states_akt.WB(t+1)='N';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
            
            % ha nem megy at I3-ba akkor lehet h meggyogyul ha min 5 napja
            % I2
         elseif  (agent_states_akt.PPsc/144)>t_min_immunity & rand<0.5
            agent_states_akt.PP(t+1)='I';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
             
         else
        
            agent_states_akt.PP(t+1)='2';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
            agent_states_akt.flag_diagnosed=0;
         end
 
elseif agent_states_akt.PP(t)=='3'
    
          % atmegy-e I4-be
        
         if age>60
             offset=0.2;
         else
             offset=0; % lehet meg tobb eset is..
         end
         
         trans_prob=probability_of_I3_I4_transition+offset;
         
         if rand<trans_prob
             
            agent_states_akt.PP(t+1)='4';
            agent_states_akt.WB(t+1)='M';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
            
            % ha nem megy at I4-ba akkor lehet h meggyogyul ha min 5 napja
            % I3
         elseif  (agent_states_akt.PPsc/144)>t_min_immunity & rand<0.5
            agent_states_akt.PP(t+1)='I';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=0;
            agent_states_akt.flag_diagnosed=0;             
             
         else
        
        
            agent_states_akt.PP(t+1)='3';
            agent_states_akt.WB(t+1)='N';
            agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
            agent_states_akt.flag_diagnosed=0;
         end
    
 elseif agent_states_akt.PP(t)=='4'
    
          % atmegy-e I5-be
        
         if age>60
             offset=0.35;
         else
             offset=0; % lehet meg tobb eset is..
         end
         
         trans_prob=probability_of_I4_I5_transition+offset;
         
         if rand<trans_prob
             
            agent_states_akt.PP(t+1)='5';
            agent_states_akt.WB(t+1)='S';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
            
            % ha nem megy at I4-ba akkor lehet h meggyogyul ha min 5 napja
            % I3
         elseif  (agent_states_akt.PPsc/144)>t_min_immunity & rand<0.5
            agent_states_akt.PP(t+1)='I';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
             
         else
        
        
            agent_states_akt.PP(t+1)='4';
            agent_states_akt.WB(t+1)='M';
            agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
            agent_states_akt.flag_diagnosed=0;
         end   
  
         
 elseif agent_states_akt.PP(t)=='5'
    
          % atmegy-e D-be
        
         if age>60
             offset=0.15;
         else
             offset=0; % lehet meg tobb eset is..
         end
         
         trans_prob=morbidity_in_I5+offset;
         
         if rand<trans_prob
             
            agent_states_akt.PP(t+1)='D';
            agent_states_akt.WB(t+1)='D';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
            
         elseif  (agent_states_akt.PPsc/144)>(t_min_immunity-1) & rand<0.5
            agent_states_akt.PP(t+1)='I';
            agent_states_akt.WB(t+1)='W';
            agent_states_akt.PPsc=0; 
            agent_states_akt.flag_diagnosed=0;             
             
         else

            agent_states_akt.PP(t+1)='5';
            agent_states_akt.WB(t+1)='S';
            agent_states_akt.PPsc=agent_states_akt.PPsc+1; 
            agent_states_akt.flag_diagnosed=0;
         end         
     
elseif agent_states_akt.PP(t)=='S' | agent_states_akt.PP(t)=='I' | agent_states_akt.PP(t)=='D'% ha 'S'-ben, D-ben vagy I-ben van, nincs Dp related update

            agent_states_akt.PP(t+1)=agent_states_akt.PP(t);
            agent_states_akt.WB(t+1)=agent_states_akt.WB(t);
            agent_states_akt.PPsc=agent_states_akt.PPsc+1;  
            agent_states_akt.flag_diagnosed=0;    
            
            
    
end

agent_states_new=agent_states_akt;

end

function y=prob_trans_from_latency(x)

p=4;  % ezekkel annak a valseget lehet beallitani hogy mennyi ido alatt er veget a lappangasi periodus
 k=p/5;
 b=2;

y=1/(1+exp((p-x)/k))^b;

end

function [day,hour,minute]=re_convert_time(t)
      % [ora,perc -> 1-144]
      
      t=t-1;
      
      day=floor(t/144);
      time_in_actual_day=mod(t,144);
      hour=floor(time_in_actual_day/6);
      minute=mod(time_in_actual_day,6)*10;
      
end
