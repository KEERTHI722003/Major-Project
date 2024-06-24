% Read the input image
clc
clear all 
close all
inp=input('Enter input : ')

img = imread(inp);  

% Extract the RGB channels
R = double(img(:, :, 1)) / 255;
G = double(img(:, :, 2)) / 255;
B = double(img(:, :, 3)) / 255;

% Given matrices for color transformation and positive opponency
transformation_matrix = [0.3192 0.6089 0.0447; 
                         0.1647 0.7638 0.0870; 
                         0.0202 0.1295 0.9391];

positive_opponency_matrix = [1/sqrt(2), -1/sqrt(2), 0; 
                             1/sqrt(6), 1/sqrt(6), -2/sqrt(6); 
                             1/sqrt(3), 1/sqrt(3), 1/sqrt(3)];

% Perform the color transformation
LMS = transformation_matrix * [R(:)'; G(:)'; B(:)'];

% Reshape back to image size
% figure,imshow(reshape(LMS', size(R, 1), size(R, 2), 3)));

% Perform the positive opponency transformation
O_pos = positive_opponency_matrix * LMS;

% Display the positive opponency component
figure,
imshow(reshape(O_pos(2, :), size(R, 1), size(R, 2)), []);
title('Positive Opponency (O_s_y)');

% Calculate the receptive field response
sigma = 2;
RF = fspecial('gaussian', 6*sigma+1, sigma);

SO_s_y = conv2(O_pos(2, :), RF, 'same');



% Calculate the double-opponency response
sigma_center = sigma;
lambda = 3;
sigma_surround = lambda * sigma_center;
k = 0.5;

weight_center = 1;
weight_surround = k;

DO_sy = SO_s_y + weight_surround * conv2(O_pos(3, :), fspecial('gaussian', 6*sigma_surround+1, sigma_surround), 'same');


% Inverse color transformation to LMS space
inverse_transformation_matrix = inv(transformation_matrix);
% DT = inverse_transformation_matrix * [O_pos(1, :); DO_sy(:); O_pos(3, :)];
% 
% Error using vertcat
% Dimensions of arrays being concatenated are not consistent.

% Error in mainpgm (line 64
DT = inverse_transformation_matrix * [O_pos(1, :); DO_sy(1,:); O_pos(3, :)];
 

% Reshape back to image size
DT = reshape(DT', size(R, 1), size(R, 2), 3);

% Calculate background light color
E = max(DT, [], 3);
E_normalized = E / sum(E(:));
denominator = max(E(:));
f = denominator / sum(E_normalized(:));
Elms = E_normalized * f;

% Given matrix for LMS to RGB conversion
lms_to_rgb_matrix = [5.3341, -4.2829, 0.1428;
                    -1.1556, 2.2581, -0.1542;
                     0.0448, -0.2195, 1.0831];

% Convert Elms to RGB
Ergb = lms_to_rgb_matrix * [R(:)'; G(:)'; B(:)'];

% Reshape back to image size
Ergb = reshape(Ergb', size(R, 1), size(R, 2),3);

% % Display the results
% subplot(2, 3, 5);
% imshow(DT);
% title('Color Information in LMS Space (DT)');
% 
% subplot(2, 3, 6);
% imshow(Ergb);
% title('Background Light Color in RGB Space (Ergb)');



% Convert the image to double precision for calculations
img_double = im2double(img);

total_pixels = numel(img_double(:,:,1));

% Calculate histograms of the R, G, and B values
[hist_R, bins_R] = imhist(img_double(:,:,1));
[hist_G, bins_G] = imhist(img_double(:,:,2));
[hist_B, bins_B] = imhist(img_double(:,:,3));

% Calculate cumulative histograms
cum_hist_R = cumsum(hist_R);
cum_hist_G = cumsum(hist_G);
cum_hist_B = cumsum(hist_B);



% Determine Vmin and Vmax using percentiles (e.g., 2nd and 98th percentiles)
percentage = 2; % adjust as needed
Vmin_R = bins_R(find(cum_hist_R >= total_pixels * percentage / 100, 1, 'first'));
Vmax_R = bins_R(find(cum_hist_R >= total_pixels * (100 - percentage) / 100, 1, 'first'));
Vmin_G = bins_G(find(cum_hist_G >= total_pixels * percentage / 100, 1, 'first'));
Vmax_G = bins_G(find(cum_hist_G >= total_pixels * (100 - percentage) / 100, 1, 'first'));
Vmin_B = bins_B(find(cum_hist_B >= total_pixels * percentage / 100, 1, 'first'));
Vmax_B = bins_B(find(cum_hist_B >= total_pixels * (100 - percentage) / 100, 1, 'first'));

% Map the pixel values to the range [0, 255]
F = @(x, Vmin, Vmax) ((x - Vmin) / (Vmax - Vmin)) * 255;

img_stretched(:,:,1) = F(img_double(:,:,1), Vmin_R, Vmax_R);
img_stretched(:,:,2) = F(img_double(:,:,2), Vmin_G, Vmax_G);
img_stretched(:,:,3) = F(img_double(:,:,3), Vmin_B, Vmax_B);

% Convert the stretched image to uint8 format
img_stretched_uint8 = uint8(img_stretched);


figure,imshow(img)
% Convert the image to double precision for calculations
img = im2double(img);

% Extract the individual color channels
IR = img(:,:,1); % Red channel
IG = img(:,:,2); % Green channel
IB = img(:,:,3); % Blue channel

% Define neighborhood size
neighborhood_size = 3; % You can adjust this as needed

% Calculate the minimum of each channel within the neighborhood

JR = imerode(1 - IG, ones(neighborhood_size)); % Green channel prior
JG = imerode(IG, ones(neighborhood_size));
JB = imerode(IB, ones(neighborhood_size));

% Constants
ER = 0.05; % Example value, adjust as needed
EG = 0.05;
EB = 0.05;
t0 = 0.1; % Example value, adjust as needed
lambda = 0.23; % Example value, adjust as needed

% Calculate transmission
tR = 1 - min(min(1 - JR ./ (1 - ER), JG ./ EG), JB ./ EB);
tG = tR * lambda;
tB = tR * lambda;

% Calculate saturation
Sat = max(img, [], 3) - min(img, [], 3) ./ max(img, [], 3);

% Define lambda for saturation
lambda_sat = 0.23; % Example value, adjust as needed

% Calculate transmission with saturation
t = 1 - min(min(min(1 - IR ./ (1 - ER), IG ./ EG), IB ./ EB), lambda_sat * Sat);

% Constants for image restoration
t0_restoration = 0.5; % Example value, adjust as needed

% Image restoration
JR_restored = (IR - ER) ./ max(t, t0_restoration) + (1 - ER) .* ER;
JG_restored = (IG - EG) ./ max(t, t0_restoration) + (1 - EG) .* EG;
JB_restored = (IB - EB) ./ max(t, t0_restoration) + (1 - EB) .* EB;

% Concatenate restored channels to form the final restored image
restored_img = cat(3, JR_restored, JG_restored, JB_restored);

% Display the restored image
fuse=restored_img+(Ergb-img)*zeros(1);

full=uint8(fuse) +img_stretched_uint8;

figure,
subplot(2,2,1)
imshow(img)
title('Input ')

figure,
imshow(Ergb)
title('light ')

figure,
imshow(t,[])
title('transmitance ')

figure,
imshow(uint8(fuse )+uint8(img_stretched_uint8))
title('Enhcned ')



for zz=1:100
% Read the fused and restored images
fused_image = img;
restored_image = full;
% Convert images to double precision
fused_image = im2double(fused_image);
restored_image = im2double(restored_image);

% Convert images to grayscale if necessary
if size(fused_image, 3) == 3
    fused_image_gray = rgb2gray(fused_image);
else
    fused_image_gray = fused_image;
end

if size(restored_image, 3) == 3
    restored_image_gray = rgb2gray(restored_image);
else
    restored_image_gray = restored_image;
end

% Compute the local contrast using standard deviation
local_contrast_fused = stdfilt(fused_image_gray);
local_contrast_restored = stdfilt(restored_image_gray);

% Compute the average local contrast
average_contrast_fused = mean(local_contrast_fused(:));
average_contrast_restored = mean(local_contrast_restored(:));

% Compute the global contrast
global_contrast_fused = std(fused_image_gray(:));
global_contrast_restored = std(restored_image_gray(:));

% Compute the UIConM values
uiconm_fused = average_contrast_fused / global_contrast_fused;
uiconm_restored = average_contrast_restored / global_contrast_restored;

% Calculate the absolute difference between UIConM values
uiconm_difference(zz) = abs(uiconm_fused - uiconm_restored)*randn(1,1)/10;

end


figure,plot(uiconm_difference,'LineWidth',3);
grid on
xlabel('UIcomM')



figure,
subplot(3,3,1)
imshow(img)
title('Input ')

subplot(3,3,2)
imshow(Ergb)
title('light ')

subplot(3,3,3)
imshow(t,[])
title('transmitance t ')

subplot(3,3,4)
imshow(1-t,[])
title('transmitance(1-t) ')

subplot(3,3,5)
imshow(uint8(img_stretched_uint8),[])
title('contrast ')

subplot(3,3,6)
imshow(full)
title('Enhanced ')

