function y=display_epidemic_curves(agent_states)

y=[];

no_agents=length(agent_states);

agent_states_akt=agent_states{1};

no_simulated_steps=length(agent_states_akt.PP)

no_simulated_days=floor(no_simulated_steps-1)/144

vect_S=[]; % 1. element: number of sgants in S on day 1
vect_I1=[];
vect_I3=[];
vect_I4=[];
vect_I5=[];
vect_R1=[];
vect_R2=[];
vect_D=[];

t=-143;

for day_index=1:no_simulated_days
    t=t+144;
    
    no_S_akt=0;
    no_I1_akt=0;
    no_I2_akt=0;
    no_I3_akt=0;
    no_I4_akt=0;
    no_I5_akt=0;
    no_R1_akt=0;
    no_R2_akt=0;
    no_D_akt=0;
    
    for i=1:no_agents;
        agent_states_akt=agent_states{i};
        
        if agent_states_akt.PP(t)=='S';
            no_S_akt=no_S_akt+1;
        elseif agent_states_akt.PP(t)=='1';
            no_I1_akt=no_I1_akt+1;
        elseif agent_states_akt.PP(t)=='2';
            no_I2_akt=no_I2_akt+1;            
        elseif agent_states_akt.PP(t)=='3';
            no_I3_akt=no_I3_akt+1;            
        elseif agent_states_akt.PP(t)=='4';
            no_I4_akt=no_I4_akt+1;            
        elseif agent_states_akt.PP(t)=='5';
            no_I5_akt=no_I5_akt+1;            
        elseif agent_states_akt.PP(t)=='I';
            no_R1_akt=no_R1_akt+1;                        
        elseif agent_states_akt.PP(t)=='R';
            no_R2_akt=no_R2_akt+1;            
        elseif agent_states_akt.PP(t)=='D';
            no_D_akt=no_D_akt+1;            
        end
    end
    
    vect_S(day_index,1)=no_S_akt;
    vect_I1(day_index,1)=no_I1_akt;
    vect_I2(day_index,1)=no_I2_akt;
    vect_I3(day_index,1)=no_I3_akt;
    vect_I4(day_index,1)=no_I4_akt;
    vect_I5(day_index,1)=no_I5_akt;
    vect_R1(day_index,1)=no_R1_akt;
    vect_R2(day_index,1)=no_R2_akt;
    vect_D(day_index,1)=no_D_akt;
end

days=[1:no_simulated_days];

figure
plot(days,vect_S,'g-','Linewidth',2)
hold
grid
plot(days,vect_I1,'b-','Linewidth',2)
plot(days,vect_I2+vect_I3+vect_I4+vect_I5,'r-','Linewidth',2)
%plot(days,vect_I1,'b-','Linewidth',2)
%plot(days,vect_I2,'r:','Linewidth',2)
%plot(days,vect_I3,'r-.','Linewidth',2)
%plot(days,vect_I4,'r--','Linewidth',2)
%plot(days,vect_I5,'r-','Linewidth',2)
plot(days,vect_R1,'m-','Linewidth',2)            
plot(days,vect_R2,'m-.','Linewidth',2)                    
plot(days,vect_D,'k-','Linewidth',2)          

legend('S','I1 - incubation','I2+I3+I4+I5','Resistant','Partially resistant','Dead')

%plot(days,vect_S + vect_I1 + vect_I2 + vect_I3 + vect_I4 + vect_I5 + vect_R1 + vect_R2 +vect_D,'k--','Linewidth',2)   
    
    
    