function [agent_states_new,L_o_A_new]=update_agents_epidemic(agents,agent_states,no_agents,L_o_A,types,t,data_locations)

%============================


for i=1:length(agent_states)
   
    agent_states_akt=agent_states{i};
    
    %==================
    
    agent_states_akt=update_agent_states(i,agents,agent_states_akt,L_o_A,data_locations,t,agent_states);
    
    %======================== ha nem valtoztatunk a state-eken
    
    %agent_states_akt.PP(t+1)=agent_states_akt.PP(t);  
    %agent_states_akt.WB(t+1)=agent_states_akt.WB(t);
    %agent_states_akt.PPsc(t+1)=0; 
    %agent_states_akt.flag_diagnosed(t+1)=0;
        
    %==========================
    
   agent_states_new{i}= agent_states_akt;
   
end
   
%============================                               
                               
L_o_A_new=update_agent_locations(agents,agent_states,no_agents,L_o_A,types,t,data_locations);
