function y=display_locations(locations,figure_index,axis_in);

y=[];

hanyat_rajzolunk_ki=length(locations); % osszeset
%hanyat_rajzolunk_ki=2;

for i=1:hanyat_rajzolunk_ki


    location_akt=locations{i};
    
    obj_centers(i,:)=location_akt.coords;
    
    area_akt=location_akt.area;
    
    %area_akt=70;
    
    width_obj(i)=determine_object_width_from_area(area_akt);        
    
end
        

        
for i=1:size(obj_centers,1);
    curve_x_obj{i}=[obj_centers(i,1)-width_obj(i)/2, obj_centers(i,1)-width_obj(i)/2, obj_centers(i,1)+width_obj(i)/2 obj_centers(i,1)+width_obj(i)/2 obj_centers(i,1)-width_obj(i)/2];
    curve_y_obj{i}=[obj_centers(i,2)-width_obj(i)/2, obj_centers(i,2)+width_obj(i)/2, obj_centers(i,2)+width_obj(i)/2 obj_centers(i,2)-width_obj(i)/2 obj_centers(i,2)-width_obj(i)/2];

end




    figure(figure_index)
    plot(curve_x_obj{1},curve_y_obj{1},'k','Linewidth',2)
    hold on
    for i=2:hanyat_rajzolunk_ki
        plot(curve_x_obj{i},curve_y_obj{i},'k','Linewidth',2)
    end
    axis(axis_in)
    



