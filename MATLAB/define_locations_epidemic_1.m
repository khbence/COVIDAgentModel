function [data_locations]=define_locations_epidemic_1

% location reference index - hol tartunk a location-ok vektoraban
lri=0;

%=========== residences=================

no_residence=30;

no_types.no_residence=no_residence;

ref_indices.ref_inex_residences=0;

for i=1:no_residence
    
    %physical coords
    x_akt=i-10*floor((i-1)/10);
    y_akt=floor((i-1)/10);
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=70;    
    
    % minimal time spent there    
    location_akt.t_min=3;
    
    locations{lri+i}=location_akt;
    
end

lri=no_residence;

%===================== schools =======

no_schools=3;

no_types.no_schools=no_schools;
ref_indices.ref_inex_schools=lri;

for i=1:no_schools
    
    %physical coords
    x_akt=2+3*(i-1);
    y_akt=-4;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=3000;    
    
    % minimal time spent there    
    location_akt.t_min=1; % azert egy, mert a szulok csak beugranak
    
    locations{lri+i}=location_akt;
    
end

lri=lri+no_schools;


%===================== small workplaces =======

no_workplace_small=7;

no_types.no_workplace_small=no_workplace_small;
ref_indices.ref_inex_workplace_small=lri;

for i=1:no_workplace_small
    
    %physical coords
    x_akt=i;
    y_akt=6;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=250;    
    
    % minimal time spent there    
    location_akt.t_min=24; % ha mar valaki bemegy min 4 orat bent van
    
    locations{lri+i}=location_akt;
    
end

lri=lri+no_workplace_small;

%===================== large workplaces =======

no_workplace_large=3;

no_types.no_workplace_large=no_workplace_large;
ref_indices.ref_inex_workplace_large=lri;

for i=1:no_workplace_large
    
    %physical coords
    x_akt=no_workplace_small + i;
    y_akt=6;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=2000;    
    
    % minimal time spent there    
    location_akt.t_min=24; % ha mar valaki bemegy min 4 orat bent van
    
    locations{lri+i}=location_akt;
    
end

lri=lri+no_workplace_large;

%===================== shops =======

no_shops=5;

no_types.no_shops=no_shops;
ref_indices.ref_inex_shops=lri;

for i=1:no_shops
    
    %physical coords
    x_akt=i*2-1;
    y_akt=4;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=randi([25,2000]);    
    
    % minimal time spent there    
    location_akt.t_min=3; % ha mar valaki bemegy min 30 percet bent van
    
    locations{lri+i}=location_akt;
    
end

lri=lri+no_shops;

%===================== city park =======

no_city_park=1;

no_types.no_city_park=no_city_park;
ref_indices.ref_inex_city_park=lri;  

    %physical coords
    x_akt=2;
    y_akt=-6;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=1000000;    
    
    % minimal time spent there    
    location_akt.t_min=6; % min 1 ora
    
    locations{lri+1}=location_akt;
    

lri=lri+no_city_park;


%===================== cinema =======

no_cinema=1;

no_types.no_cinema=no_cinema;
ref_indices.ref_inex_cinema=lri;  

    %physical coords
    x_akt=4;
    y_akt=-6;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=1200;    
    
    % minimal time spent there    
    location_akt.t_min=12; % min 2 ora
    
    locations{lri+1}=location_akt;


lri=lri+no_cinema;


%===================== health_centre =======

no_health_centre=1;

no_types.no_health_centre=no_health_centre;
ref_indices.ref_inex_health_centre=lri;  

    %physical coords
    x_akt=6;
    y_akt=-6;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=600;    
    
    % minimal time spent there    
    location_akt.t_min=4; % min 3/4 ora
    
    locations{lri+1}=location_akt;
    

lri=lri+no_health_centre;


%===================== hospital =======

no_hospital=1;

no_types.no_hospital=no_hospital;
ref_indices.ref_inex_hospital=lri;  

    %physical coords
    x_akt=8;
    y_akt=-6;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=6000;    
    
    % minimal time spent there    
    location_akt.t_min=6*24*3; % min 3 nap 
    
    locations{lri+1}=location_akt;
    

lri=lri+no_hospital;

%===================== environment =======

    no_environmental_compartments=1;

    no_types.no_environmental_compartments=no_environmental_compartments;
    ref_indices.ref_index_environmental_compartments=lri;  
    
        %physical coords
    x_akt=5;
    y_akt=-2;
        
    location_akt.coords=[x_akt,y_akt];
    
    % area [m^2]
    location_akt.area=20000; 
    
    location_akt.t_min=1;
    
    locations{lri+1}=location_akt;
    
%======================    


    data_locations.locations=locations;
    data_locations.no_types_locations=no_types;
    data_locations.ref_indices_locations=ref_indices;
