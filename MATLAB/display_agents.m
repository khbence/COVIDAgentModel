function y=display_agents(locations,agents,agent_states,L_o_A,t,figure_index,axis_in);

y=[];

type_1_symbol='o';
type_2_symbol='d';

state_S_color='g';  % S
state_I1_color='b'; %incubation period
state_I2_color='y'; % no symptoms
state_I3_color='r'; % mild symptoms
state_I4_color='r'; % moderate symptoms
state_I5_color='r'; % serious symptoms
state_R1_color='m'; % immune to infection - R1
state_R2_color='m'; % reduced immunity - R2
state_D_color='k'; % dead - D


location_indices_akt=L_o_A(:,t);

n_A=length(location_indices_akt);

figure(figure_index)

for i=1:n_A
    agent_akt=agents{i};
    type_akt=agent_akt.type;
    
    if type_akt==1
        symbol=type_1_symbol;
    elseif type_akt==2
        symbol=type_2_symbol;
        %...
    end
    
    agent_states_akt=agent_states{i};
    
    state_PP_akt=agent_states_akt.PP(t);
    
    if state_PP_akt=='S'
        color=state_S_color;
    elseif state_PP_akt=='1' % I1 kodja =1
        color=state_I1_color;
    elseif state_PP_akt=='2'
        color=state_I2_color;    
    elseif state_PP_akt=='3'
        color=state_I3_color;        
    elseif state_PP_akt=='4'
        color=state_I4_color;
    elseif state_PP_akt=='5'
        color=state_I5_color;        
    elseif state_PP_akt=='I'
        color=state_R1_color;        
    elseif state_PP_akt=='R'
        color=state_R2_color;
    elseif state_PP_akt=='D'
        color=state_D_color;    
    end
   
    location_index_akt=location_indices_akt(i,1);
    
    location_akt=locations{location_index_akt};
    
    coords_akt=location_akt.coords;
    
    %kicsit megperturbaljuk hogy latszodjon ha tobbenvannak 1 helyen.
    
    area_akt=location_akt.area;
    
    width_obj_akt=determine_object_width_from_area(area_akt);        
    
    x_offset=1.4*(rand-0.5)*width_obj_akt;
    y_offset=1.4*(rand-0.5)*width_obj_akt;
    
    coords_akt_2_display=coords_akt+[x_offset, y_offset];
    
    plot(coords_akt_2_display(1),coords_akt_2_display(2),[color symbol])
end
    
    
