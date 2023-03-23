%% CM2 related code on forward simulation with scattering value noise

pwdd = pwd;
addpath('utils');
% useful functions
VEC = @(x) x(:);
clip = @(x, vmin, vmax) max(min(x, vmax), vmin);
F2D = @(x) fftshift(fft2(ifftshift(x)));
Ft2D = @(x) fftshift(ifft2(ifftshift(x)));
pad2d = @(x) padarray(x,round(0.5*size(x)));
crop2d = @(x) x(1+size(x,1)/4:size(x,1)/4*3,1+size(x,2)/4:size(x,2)/4*3);
conv2d = @(obj,psf) crop2d(real(Ft2D(F2D(pad2d(obj)).*F2D(pad2d(psf)))));
unit_norm = @(x) x./norm(x(:));
auto_corr = @(x) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(x))))));
x_corr = @(x,y) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(y))))));
linear_normalize = @(x) (x - min(x(:)))./(max(x(:))-min(x(:)));

xsen = 2076; ysen = 3088; %num pixels of sensor

%% free space psf
load('cPSF.mat')
psfs = psf(:,:,end-24+1:end);
clear psf


%% synthesize a single 2D measurement by depth-wise convolution between
[rows,cols,depth] = size(psfs);
[xx,yy] = meshgrid([-cols/2:1:cols/2-1], [-rows/2:1:rows/2-1]);%% coordinates of full fov in pixel


Nss = 22;
[sx,sy,sz] = meshgrid([-Nss:Nss].*(1/5), [-Nss:Nss].*(1/5), [-2:2]);%% coordinates of a small sphere volume
mag = 52/90;
particle_size = randn*2+15;
obj_space_sampling = 4.15; %.53/2*9/mag;

fdiam = 512;
% layer 21 in og psf, center of foci for each microlens
psfloc40 = [406,909; 407,1545; 405,2175; 1037,911; 1037,1544; 1037,2176; 1675,911; 1675,1543; 1675,2173]; 

FOVrad = 2000; %2mm FOV
FOVpix = round(FOVrad/obj_space_sampling/2); %pixel for radius
bglog = zeros(rows,cols);
for ww = 1:9
    w22 = psfloc40(ww,1); w11 = psfloc40(ww,2);
    bglog(sqrt((yy-yy(w22,1)).^2+(xx-xx(1,w11)).^2)<=(fdiam)/2) = 1;
end
bglog =  imgaussfilt(bglog,100);
fdiammesh = 600;
[xb,yb] = meshgrid(-fdiammesh/2:fdiammesh/2-1,-fdiammesh/2:fdiammesh/2-1);
bglog2= sqrt((xb).^2+(yb).^2)<=fdiam/2;
bglog2 = imgaussfilt(double(bglog2),40);
clear xb yb
% aa = imerode(y,se);
% bg = imdilate(aa,se);
trainingdata = 'H:\jeffrey\scattering\training data\dataset11_1500';
if ~exist([trainingdata], 'dir')
       mkdir([trainingdata])
end
if ~exist([trainingdata,'\gt'], 'dir')
       mkdir([trainingdata,'\gt'])
end
if ~exist([trainingdata,'\rfv'], 'dir')
       mkdir([trainingdata,'\rfv'])
end
if ~exist([trainingdata,'\rfvbg'], 'dir')
       mkdir([trainingdata,'\rfvbg'])
end
if ~exist([trainingdata,'\stack'], 'dir')
       mkdir([trainingdata,'\stack'])
end
if ~exist([trainingdata,'\stackbg'], 'dir')
       mkdir([trainingdata,'\stackbg'])
end


paramlist = zeros(1000,2);

% z = 0:25:600-25;
% psfsd = zeros(size(psfs));
% 
% d = exp(-z/600);
% d = d/max(d);
% 
% for i = 1 : depth
%     psfsd(:,:,i) = psfs(:,:,i).* d(i);
% end


%%
clc
tic
wc2 = 1039; wc1 = 1545; depth2 = 24;
mean_bg = 0.5; fdiam2 = 512;
ls = inf;
for ii = 1:1500 % 
    
    ii
    fdiam = 600;

    num_particles = round(rand*200+300);
%     paramlist(ii,2) = num_particles;
    % COMPUTE GROUND TRUTH
    gt_volume = zeros(rows,cols,depth);
    gt_locations = zeros(num_particles,4);
    std_lum = 0.1;
    for i = 1:num_particles
        particle_size = randn*2 + 15;
% %         particle_size = 17;
        current_radius = (particle_size/obj_space_sampling/2);
    %     paramlist(ii,1) = current_radius;
        sphere = zeros(size(sx)); % generate a dense sphere volume
        sphere(sqrt(sx.^2+sy.^2+sz.^2)<=current_radius) = 1;
        %
        ab = size(sx,1)/5;
        sphere2 = zeros(ab,ab,5);
        for ie = 1:5
            sphere2(:,:,ie) = average_shrink(sphere(:,:,ie),5);
        end
        sphere = sphere2;
        sphere = sum(sphere,3);

        zz = randi([2, depth-1]);
        while true
            rr = randi([9,rows-9]);
            cc = randi([9,cols-9]);
            if sqrt(xx(rr,cc).^2 + yy(rr,cc).^2) <= FOVpix  % for cylinder radius,241 pixels is 2mm in diameter
                break
            end
        end

        
        gt_locations(i,1) = rr;
        gt_locations(i,2) = cc;
        gt_locations(i,3) = zz;

        tmp_lum = .8 + std_lum*randn(1);
        tmp_lum = clip(tmp_lum, 0,1);
        gt_locations(i,4) = tmp_lum;
        q = tmp_lum * sphere;
%         q = clip(q,2,8); % the OG
        q = clip(q,2,8);
        q(q<=2) = 0;
        gt_volume(rr-4:rr+4,cc-4:cc+4,zz)=gt_volume(rr-4:rr+4,cc-4:cc+4,zz)+q;
    end 
%     gt_volume = clip(gt_volume,0,5); % the og
    gt_volume = clip(gt_volume,0,5);
    cropgt = gt_volume(wc2-fdiam/2:wc2+fdiam/2-1, wc1-fdiam/2:wc1+fdiam/2-1,:);
    cropgt = cropgt(fdiam/2-fdiam2/2:fdiam/2+fdiam2/2-1,fdiam/2-fdiam2/2:fdiam/2+fdiam2/2-1,:); 
    
    % scattering attenuation
%     y_partd = gather(cm2_forward_gpu(gt_volume,psfsd,false)); % can change to CPU version
%     y_partd = clip(y_partd,0,1470000);
%     yd = y_partd;
%     yd = yd./max(yd(:));
    
    % free space
    y_part1 = gather(cm2_forward_gpu(gt_volume,psfs,false)); % can change to CPU version
    y_part1 = clip(y_part1,0,1470000);
    y = y_part1;
    y = y./max(y(:));
    
    %%%%% for mean value finding of beads (use attenuation convolution for
    %%%%% this calculation
    ytest = y(wc2-fdiam/2:wc2+fdiam/2-1, wc1-fdiam/2:wc1+fdiam/2-1,:);
    ytest(ytest<0.25) = 0;
    ind = imregionalmax(ytest);
    vals = ytest(ind);
    mean_signal = mean(vals); % THIS IS "S" FROM THE PAPER
%     histogram(vals,'numbins',100)
    %%%%%
    value = im2double(imread(['H:\jeffrey\scattering\training data\valuesamples2\','value_',num2str(ii),'.png']));
%     value = im2double(imread(['E:\jeffrey\cm2 dl\student teacher\training data\value samples\','value_',num2str(randi([1001,2000])),'.tif']));
%     value = imresize(value,[600,600]);
    value = value2D(600);
    mean_bg = mean(value(:));
    value = bglog2.*value;
    
    fdiam = 600;
    bgmask = zeros(size(bglog));

    for ww = 1 : 9
        w22 = psfloc40(ww,1); w11 = psfloc40(ww,2);
        bgmask(w22-fdiam/2:w22+fdiam/2-1,w11-fdiam/2:w11+fdiam/2-1) = value.*bglog2;
    end
    bgmask = bgmask.*bglog;

    unifa = 1.1; unifb = 3; %0.00625: 1.01; 0.03125: 1.05; 0.0625: 1.1; 0.9375: 2.5
    SBR = (rand*(unifb-unifa)+unifa); %Unif[unifa,unifb]
    SBR = 1.08;
%     S = 0.9375;
    S = (mean_bg*SBR-mean_bg)/mean_signal;
%     SBR = (S*0.8+.5)/.5;
    paramlist(ii,:) = [num_particles, SBR];
    bgmask = linear_normalize(bgmask);
    y = linear_normalize(y);
%     yd = linear_normalize(yd);
    ybg = linear_normalize(S*y+bgmask); % with the SBR that i want

    ybgstack = [];
    ystack = [];
    for ww = 1:9
            
        w2 = psfloc40(ww,1); w1 = psfloc40(ww,2);
        
        aaa = (y(w2-fdiam/2:w2+fdiam/2-1,...
                w1-fdiam/2:w1+fdiam/2-1));
        aaa = linear_normalize(aaa);
        ystack = cat(3,ystack,(aaa));

        aaa = (ybg(w2-fdiam/2:w2+fdiam/2-1,...
                w1 - fdiam/2:w1 + fdiam/2-1));
        aaa = linear_normalize(aaa);
        ybgstack = cat(3,ybgstack,(aaa));

    end


    for j = 1 : 9
        if j == 5
            continue
        end
        [ystack(:,:,j),shift] = register_single_psf(ystack(:,:,5),ystack(:,:,j),128);
        ybgstack(:,:,j) = imtranslate(ybgstack(:,:,j),shift);
    end

    ystack = ystack(fdiam/2-fdiam2/2:fdiam/2+fdiam2/2-1,fdiam/2-fdiam2/2:fdiam/2+fdiam2/2-1,:);
    ybgstack = ybgstack(fdiam/2-fdiam2/2:fdiam/2+fdiam2/2-1,fdiam/2-fdiam2/2:fdiam/2+fdiam2/2-1,:);
    
    len = fdiam2; 
%     lf = permute(reshape(ystack,[len,len,3,3]), [1,2,4,3]);
%     rfv = zeros(len, len, depth2);
    shiftt = 13;
%     for j = 1:depth2
%         rfv(:,:,j) = lf_refocusing(lf, j-shiftt);
%     end
%     rfv = rfv ./ max(rfv(:));
%     rfv = 255*rfv;
    
    lf = permute(reshape(ybgstack,[len,len,3,3]), [1,2,4,3]);
    rfvbg = zeros(len, len, depth2);
    for j = 1:depth2
        rfvbg(:,:,j) = lf_refocusing(lf, j-shiftt);
    end
    rfvbg = rfvbg ./ max(rfvbg(:));
    rfvbg = 255*rfvbg;
    
    cd(trainingdata);
%     cd 'stack'
%     write_mat_to_tif(uint8(255*linear_normalize(ystack)),['sim_meas_',num2str(ii-1),'.tif']);
%     cd(trainingdata);
%     
    cd 'stackbg'
    write_mat_to_tif(uint8(255*linear_normalize(ybgstack)),['sim_meas_',num2str(ii-1),'.tif']);
    cd(trainingdata);

%     cd 'rfv'
%     write_mat_to_tif(uint8((rfv)),['sim_meas_',num2str(ii-1),'.tif']);clc
%     cd(trainingdata);
    
    cd 'rfvbg'
    write_mat_to_tif(uint8((rfvbg)),['sim_meas_',num2str(ii-1),'.tif']);
    cd(trainingdata);

    cd 'gt'
    write_mat_to_tif(uint8(255*linear_normalize(cropgt)), ['sim_gt_vol_',num2str(ii-1),'.tif']);
    cd(pwdd)

end
toc
cd(trainingdata)
save('param_list.mat','paramlist')
%% got generating data for BGR-net
clip = @(x, vmin, vmax) max(min(x, vmax), vmin);
se = strel('disk',5,0);
depth2 = 24;
% ls = [80,160,320];
trainingdata = 'H:\jeffrey\scattering\training data\dataset11\';

parfor iw = 1:500
    aaa = 1.49e-4 + randn*5.7092e-6;
    bbb = clip(5.41e-6 + randn*2.7754e-6,0,1);
    fn = [trainingdata,'stackbg\sim_meas_',num2str(iw-1),'.tif'];
    a = im2double(read_tif_to_mat(fn,false));
    a = a + sqrt(aaa*a + bbb) .* randn(size(a));
    
    % BACKGROUND REMOVAL STEP. just 3 lines.
    aa = imerode(a,se);
    bg = imdilate(aa,se);
    a = linear_normalize(a-bg);

    len = 512; 
    lf = permute(reshape(a,[len,len,3,3]), [1,2,4,3]);
    rfv = zeros(len, len, depth2);
    shiftt = 13;
    for j = 1:depth2
        rfv(:,:,j) = lf_refocusing(lf, j-shiftt);
    end
    rfv = rfv ./ max(rfv(:));
    rfv = 255*rfv;

    toname = [trainingdata,'stackbgr\','sim_meas_',num2str(iw-1),'.tif'];
    write_mat_to_tif(uint8(255*(a)),toname);
    toname = [trainingdata,'rfvbgr\','sim_meas_',num2str(iw-1),'.tif'];
    write_mat_to_tif(uint8((rfv)),toname);

end

%%
% save 3D mat as tiff file wihtout compression
function write_mat_to_tif(mat,filename)

% save angiogram segmentation as multi-image tiff file
imwrite(mat(:,:,1),filename,'compression','none');
[~,~,nz] = size(mat);
for i = 2:nz
   imwrite(mat(:,:,i),filename,'compression','none','writemode','append'); 
end

end

function [reg_view,shift] = register_single_psf(psf0,psf2reg,max_shift)
F2D = @(x) fftshift(fft2(ifftshift(x)));
Ft2D = @(x) fftshift(ifft2(ifftshift(x)));
pad2d = @(x) padarray(x,0.5*size(x));
crop2d = @(x) x(1+size(x,1)/4:size(x,1)/4*3,1+size(x,2)/4:size(x,2)/4*3);
unit_norm = @(x) x./norm(x(:));
x_corr = @(x,y) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(y))))));
[rows,cols] = size(psf0);
center_row = rows/2 + 1;
center_col = cols/2 + 1;
sub_center_row = max_shift + 1;
sub_center_col = max_shift + 1;
xcorr_map = x_corr(psf0,psf2reg);
sub_xcorr_map = xcorr_map(center_row-max_shift:center_row+max_shift-1,...
  center_col-max_shift:center_col+max_shift-1);
[~,idx] = max(sub_xcorr_map(:));
[maxrow,maxcol] = ind2sub(size(sub_xcorr_map), idx);
shift_x = maxcol - sub_center_col;
shift_y = maxrow - sub_center_row;
shift = [shift_x, shift_y];
reg_view = imtranslate(psf2reg,shift);
end


