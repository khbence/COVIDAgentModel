function L_o_A_new=update_agent_locations(agents,agent_states,no_agents,L_o_A,types,t,data_locations);

locations=data_locations.locations;
ref_indices_locations=data_locations.ref_indices_locations;

L_o_A_new=L_o_A;

no_agents=size(L_o_A,1); % annyi agens van ahany sora van L_o_A-nak

flag_weekend=weekend_or_not_epidemic(t);

ref_index_environmental_compartments=ref_indices_locations.ref_index_environmental_compartments;


for agent_index=1:no_agents;
    
    % ez az egesz csak akkor kell, ha meg nincs beirva a t+1 elem (lehet hogy regebben meghataroztuk hogy hol lesz a legkozelebbi periodusban)
    if L_o_A_new(agent_index,t+1)==0
    
        agent_akt=agents{agent_index};
        agent_states_akt=agent_states{agent_index};

        location_index_akt=L_o_A(agent_index,t);

        type_akt= types{agent_akt.type};

        WB_state_akt=agent_states_akt.WB(t);

        if WB_state_akt=='W'
            index_1=1;
        elseif WB_state_akt=='N'
            index_1=2;
        elseif WB_state_akt=='M'        
            index_1=3;
        elseif WB_state_akt=='S'        
            index_1=4;
        elseif WB_state_akt=='D'        
            index_1=5;            
        end

        index_2=flag_weekend+1; % 2 ha weekend

        if agent_akt.type==1 % minden tipusra kulon kell sajnos megirni az update-et :(

            destination_index_akt=update_agent_type_1_location(agent_akt,type_akt,L_o_A,types,t,locations,index_1,index_2);

        % elseif agent_akt.type==2
        %...

        end
        
        % ha nem az a destination-je ahol eppen van, lefixaljuk az
        % elkovetkezo location -jait az utnak es annakmegfeleloen hogy
        % megerkezes utan mi a minimum destination-on eltoltott ido
        if location_index_akt ~= destination_index_akt
             location_akt=locations{location_index_akt};
             destination=locations{destination_index_akt};

            position_akt=location_akt.coords;
            position_dest=destination.coords;

            diff_akt=position_akt-position_dest;
            
            dist_akt=norm(diff_akt); 
            
            % ezt kesobb lehet bonyolitani
            % k az ut megtetelehez szukseges ido
            % itt most egesz egyszeruen
            k=max(1,floor(dist_akt^(1/3)));
            
            environment_index_akt=ref_index_environmental_compartments+1; % lehet hogy kesobb tobb env. kompartment lesz
            
            for i=1:k
                L_o_A_new(agent_index,t+i)=environment_index_akt;
            end
            
            min_time_to_spend_at_destination=destination.t_min;
            
            for i=1:min_time_to_spend_at_destination
                L_o_A_new(agent_index,t+k+i)=destination_index_akt; % elvileg nem loghat ki a napbol ha jol mukodik minden, bar vegul is az sem baj. 
                %A baj az ha T_sim meretén tullog (mert L_o_A-nak ennyi oszlopa van) - de ejjel mindenkinek a
                %home a destination, es 24 orakor mar egy ideje ott van
                %szoval alvileg nem lesz gaz, ha minden szimulacio 24
                %orakor vegzodik (teljes napokat szimulalunk)
            end
        else
            L_o_A_new(agent_index,t+1)=destination_index_akt;
        end
        
    end
    
end
    