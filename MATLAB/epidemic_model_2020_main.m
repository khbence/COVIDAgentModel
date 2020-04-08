function out=epidemic_model_2020_main

clc
out=[];

global virulency 

global probability_of_not_getting_sick_when_infected  % this applies if the disease did not progressed in t_min_immunity days of incubation 

global probability_of_I2_I3_transition

global probability_of_I3_I4_transition

global probability_of_I4_I5_transition

global morbidity_in_I5

global t_min_immunity



global no_infections_at_locations

virulency=0; % [in -0.3 0.2] % greater -> more virulent

probability_of_not_getting_sick_when_infected=0.5; 

probability_of_I2_I3_transition=0.5; % ez a baseline, ezt moderalja a kor

probability_of_I3_I4_transition=0.3; % ez a baseline, ezt moderalja a kor

probability_of_I4_I5_transition=0.05; % ez a baseline, ezt moderalja a kor

morbidity_in_I5=0.01;

t_min_immunity=4;


cd 'c:\matlabdata\epidemic\'

% test of time format-convsersion
% t_test_1=[13,40]
% t_test_2=convert_time(t_test_1)
% [day,hour,minute]=re_convert_time(t_test_2);
% [hour,minute]
%return

[data_locations]=define_locations_epidemic_1;

locations=data_locations.locations;

no_locations=length(locations);

no_infections_at_locations=zeros(no_locations,1);

figure_index=1;

axis=[0 11 -8 8];


%return

types=define_types_epidemic;

no_dyas_2_sim=40;

t_final=convert_time([24*no_dyas_2_sim,0]);

T_sim=t_final; % simulation time - 1 day is 144 units

[agents,agent_states,no_agents,L_o_A]=define_agents_epidemic(data_locations,T_sim);

%return

for t=1:t_final
    
    [agent_states_new,L_o_A_new]=update_agents_epidemic(agents,agent_states,no_agents,L_o_A,types,t,data_locations);

    % ebben a lepesben van a valodi update
    agent_states=agent_states_new;
    L_o_A=L_o_A_new;
    
   [day_akt,hour_akt,minute_akt]=re_convert_time(t);
   
   t_olvashato(1,t)=day_akt;
   t_olvashato(2,t)=hour_akt;
   t_olvashato(3,t)=minute_akt;

end


disp('number of infections at each location')

no_infections_at_locations

y=display_epidemic_curves(agent_states)


return
%========================= display



start_time_of_disp=[0,0];
end_time_of_disp=[24*no_dyas_2_sim,0];

%================

flag_video=0;


if flag_video==1
 writerObj = VideoWriter('epidemic_example.avi');
 writerObj.FrameRate = 5;
 open(writerObj)
end
%========

%for t=1:T_sim
for t=convert_time(start_time_of_disp):convert_time(end_time_of_disp)
       
    if mod(t,24)==0
    
    figure(figure_index)
    hold off        
        
        y=display_locations(locations,figure_index,axis);
    
        display_agents(locations,agents,agent_states,L_o_A,t,figure_index,axis);

   [day_akt,hour_akt,minute_akt]=re_convert_time(t);
   disp([num2str(day_akt) ':' num2str(hour_akt) ':' num2str(minute_akt)])        
       
    if flag_video==0 & mod(t,144)==0
        pause
    end
    
    if flag_video==1
        frame=getframe;
        writeVideo(writerObj,frame);
    end
    %pause(0.1)
    
    

   end
   
end

%L_o_A(:,convert_time(start_time_of_disp):convert_time(end_time_of_disp))

cd 'c:\matlabfiles\epidemic\'

end % end of the main function

%===============

function t_out=convert_time(t_vect_in)
      % [ora,perc -> 1-144]
      
      t_out=[(t_vect_in(1)*60 + t_vect_in(2))/10]+1;
      
end


function [day,hour,minute]=re_convert_time(t)
      % [ora,perc -> 1-144]
      
      t=t-1;
      
      day=floor(t/144);
      time_in_actual_day=mod(t,144);
      hour=floor(time_in_actual_day/6);
      minute=mod(time_in_actual_day,6)*10;
      
end

