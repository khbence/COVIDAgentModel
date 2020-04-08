function agent_states_out=update_agent_states(agent_index,agents,agent_states_akt,L_o_A,data_locations,t,agent_states)

location_index_akt=L_o_A(agent_index,t);

locations=data_locations.locations;

location_akt=locations{location_index_akt};

area_akt=location_akt.area;

agent_akt=agents{agent_index};

age=agent_akt.age;
flag_chronic=agent_akt.flag_chronic;



% disease progression related update csak minden nap 0 orakor.
% (ekkor nincs infection)
if mod(t,144)==0
   % disp([' t = ' num2str(t) ' disease progression realted update'])
    
    agent_states_new=disease_progression_related_agent_state_update(agent_index,agents,agent_states_akt,L_o_A,data_locations,t,agent_states);
else
    % infection related update minden periodusban
    agent_states_new=infection_related_agent_state_update(agent_index,agents,agent_states_akt,L_o_A,data_locations,t,agent_states);
end

agent_states_out=agent_states_new;


