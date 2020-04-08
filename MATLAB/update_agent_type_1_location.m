function destination_akt=update_agent_type_1_location(agent_akt,type_akt,L_o_A,types,t,locations,index_1,index_2)

days_index=floor((t-1)/144);

t_inside_the_day=t-days_index*144;

home=agent_akt.type_related_pars.home;
school=agent_akt.type_related_pars.school;
workplace=agent_akt.type_related_pars.workplace;
preferred_shop_1=agent_akt.type_related_pars.preferred_shop_1;
preferred_shop_2=agent_akt.type_related_pars.preferred_shop_2;
preferred_cinema=agent_akt.type_related_pars.preferred_cinema;
preferred_city_park=agent_akt.type_related_pars.preferred_city_park;
preferred_visit_site_1=agent_akt.type_related_pars.preferred_visit_site_1;
preferred_doctor=agent_akt.type_related_pars.preferred_doctor;
preferred_hospital=agent_akt.type_related_pars.preferred_hospital;
        
% ez a resz az adot tipus karakterisztikus valseg tablai alapjan
% toltodik:
relevant_prob_matrix_akt=type_akt{index_1,index_2};
        
if index_1==1
    if index_2==1;
                
        % itt pl (see define_type_1_epid)
                
        % 1. sor - otthon
        % 2. sor - iskola
        % 3. sor - munkahely
        % 4. sor - uzlet 1
        % 5. sor - uzlet 2
        alternatives_akt=[home, school, workplace,  preferred_shop_1,  preferred_shop_2];
                
        n_akt=length(alternatives_akt);
        w_akt=relevant_prob_matrix_akt(:,t_inside_the_day);
              
        choice_index_akt=randsample(n_akt,1,'true',w_akt);  % na szerintem ez lassitja majd le, biztos lehet ezt okosabban

        destination_akt= alternatives_akt(choice_index_akt);
        
    elseif index_2==2;
                
         % 1. sor - otthon
         % 2. sor - uzlet 1
         % 3. sor - uzlet 2
         % 4. sor - city_park (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
         % 5. sor - cinema (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
         % 6 sor - latogatas
        alternatives_akt=[home, preferred_shop_1,  preferred_shop_2, preferred_city_park, preferred_cinema, preferred_visit_site_1];
                
        n_akt=length(alternatives_akt);
        w_akt=relevant_prob_matrix_akt(:,t_inside_the_day);
              
        choice_index_akt=randsample(n_akt,1,'true',w_akt);  % na szerintem ez lassitja majd le, biztos lehet ezt okosabban

        destination_akt= alternatives_akt(choice_index_akt);
        
    end
  
elseif index_1==2
         if index_2==1;
                 
         % itt pl (see define_type_1_epid)
                 
        % 1. sor - otthon
        % 2. sor - iskola
        % 3. sor - munkahely
        % 4. sor - uzlet 1
        % 5. sor - uzlet 2
         alternatives_akt=[home, school, workplace,  preferred_shop_1,  preferred_shop_2];
                 
         n_akt=length(alternatives_akt);
         w_akt=relevant_prob_matrix_akt(:,t_inside_the_day);
               
         choice_index_akt=randsample(n_akt,1,'true',w_akt);  % na szerintem ez lassitja majd le, biztos lehet ezt okosabban
 
         destination_akt= alternatives_akt(choice_index_akt);
         
      elseif index_2==2;
        %                 
        % 1. sor - otthon
        % 2. sor - uzlet 1
        % 3. sor - uzlet 2
        % 4. sor - city_park (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
        % 5. sor - cinema (ezt a location-t majd ugy allitjuk be hogy a minimalis tartozkodasi ido legyen par ora)
         alternatives_akt=[home, preferred_shop_1,  preferred_shop_2, preferred_city_park, preferred_cinema];
                 
         n_akt=length(alternatives_akt);
         w_akt=relevant_prob_matrix_akt(:,t_inside_the_day);
               
         choice_index_akt=randsample(n_akt,1,'true',w_akt);  % na szerintem ez lassitja majd le, biztos lehet ezt okosabban
 
         destination_akt= alternatives_akt(choice_index_akt);
         
     end
        
elseif index_1==3
             if index_2==1;
       
            % 1. sor - otthon
            % 2. sor - iskola
            % 3. sor - munkahely
            % 4. sor - orvos
          alternatives_akt=[home, school, workplace,  preferred_doctor];
                  
          n_akt=length(alternatives_akt);
          w_akt=relevant_prob_matrix_akt(:,t_inside_the_day);
                
          choice_index_akt=randsample(n_akt,1,'true',w_akt);  % na szerintem ez lassitja majd le, biztos lehet ezt okosabban
  
          destination_akt= alternatives_akt(choice_index_akt);
          
       elseif index_2==2;
        % 1. sor - otthon
        % 2. sor - doctor
          alternatives_akt=[home, preferred_doctor];
                  
          n_akt=length(alternatives_akt);
          w_akt=relevant_prob_matrix_akt(:,t_inside_the_day);
                
          choice_index_akt=randsample(n_akt,1,'true',w_akt);  % na szerintem ez lassitja majd le, biztos lehet ezt okosabban
  
          destination_akt= alternatives_akt(choice_index_akt);
          
      end
    
elseif index_1==4    
    destination_akt=preferred_hospital;
    
elseif index_1==5    
    destination_akt=preferred_hospital;    
    
end
    
        
    
    