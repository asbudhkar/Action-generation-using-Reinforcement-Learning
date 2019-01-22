files = dir('Clapping/nturgb+d_skeletons_clapping/*.skeleton');
iter=1;
for file=files'
    try
    fileid = fopen(strcat('Clapping/nturgb+d_skeletons_clapping/',file.name));
    file.name
    framecount= fscanf(fileid,'%d',1); % no of the recorded frames
    store=[];
    store2d=[];
    for f=1:framecount
        bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
        for b=1:bodycount
            clear body;
            body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
            arrayint = fscanf(fileid,'%d',6); % read 6 integers
            lean = fscanf(fileid,'%f',2);
            body.trackingState = fscanf(fileid,'%d',1);

            body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
            for j=1:body.jointCount
                jointinfo = fscanf(fileid,'%f',11);
                store=[store;[jointinfo(1),jointinfo(2),jointinfo(3)]];
                store2d =[store2d; [jointinfo(6),jointinfo(7)]];
                joint.trackingState = fscanf(fileid,'%d',1);
            end
        end
    end
    fclose(fileid);
    k = 1;
    storenew = [];

        neck = [store(k+2,1), store(k+2,2), store(k+2,3)];
        lh = [store(k+12,1), store(k+12,2), store(k+12,3)];
        rh = [store(k+16,1), store(k+16,2), store(k+16,3)];
        u1= lh - neck;
        u1_bar = u1/norm(u1);
        u2_temp = rh - neck;
        u3 = cross(u1_bar,u2_temp)/norm(cross(u1_bar,u2_temp));
        u2 = cross(u1,u3);
        s = norm(u1);
        for j = k:(24+k)
            storenew = [storenew ;[dot(store(j,:)-neck,u1_bar)/s, dot(store(j,:)-neck,u2)/s, dot(store(j,:)-neck,u3)/s],0,0,0,0,0,0];
        end

        k=k+25;
        neck = [store(k+2,1), store(k+2,2), store(k+2,3)];
        lh = [store(k+12,1), store(k+12,2), store(k+12,3)];
        rh = [store(k+16,1), store(k+16,2), store(k+16,3)];
        u1= lh - neck;
        u1_bar = u1/norm(u1);
        u2_temp = rh - neck;
        u3 = cross(u1_bar,u2_temp)/norm(cross(u1_bar,u2_temp));
        u2 = cross(u1,u3);
        s = norm(u1);
        for j = k:(24+k)
            t=[dot(store(j,:)-neck,u1_bar)/s, dot(store(j,:)-neck,u2)/s, dot(store(j,:)-neck,u3)/s];
            u=[storenew(j-25,1), storenew(j-25,2), storenew(j-25,3)];
            v=[storenew(j-25,1)-t(1), storenew(j-25,2)-t(2), storenew(j-25,3)-t(3)];
            storenew = [storenew ;t,v,0,0,0];
        end
        k=k+25;


    for i = 3:framecount
        neck = [store(k+2,1), store(k+2,2), store(k+2,3)];
        lh = [store(k+12,1), store(k+12,2), store(k+12,3)];
        rh = [store(k+16,1), store(k+16,2), store(k+16,3)];
        u1= lh - neck;
        u1_bar = u1/norm(u1);
        u2_temp = rh - neck;
        u3 = cross(u1_bar,u2_temp)/norm(cross(u1_bar,u2_temp));
        u2 = cross(u1,u3);
        s = norm(u1);
        for j = k:(24+k)
            t=[dot(store(j,:)-neck,u1_bar)/s, dot(store(j,:)-neck,u2)/s, dot(store(j,:)-neck,u3)/s];
            a=[storenew(j-25,1)-t(1)-v(1), storenew(j-25,2)-t(2)-v(2), storenew(j-25,3)-t(3)-v(3)];
            v=[storenew(j-25,1)-t(1), storenew(j-25,1)-t(1), storenew(j-25,1)-t(1)];
            storenew = [storenew ;t,v,a];   
        end
        k=k+25;
    end
    k=1;
    storenew2=[];
    
    for i = 1:framecount
        storage=[];
        for j = k:(24+k)
            storage=[storage,storenew(j,1),storenew(j,2),storenew(j,3),storenew(j,4),storenew(j,5),storenew(j,6),storenew(j,7),storenew(j,8),storenew(j,9)];
        end
        storenew2=[storenew2;storage];
        k=k+25;
    end
    %xlswrite(strcat('output/',file.name, '.xls'),storenew);
    iter=iter+1;
    csvwrite(strcat('output2/file_',num2str(iter), '.csv'),storenew2);
    tp=0;
    
    catch err
       something=0; 
    end
end