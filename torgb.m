% Script to put gray images in color format
close all, clear all, clc;
% Treating benign
D = 'data_png/benign/';
S = dir(fullfile(D,'*.png')); % pattern to match filenames.
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    I = imread(F);
    rgbImage = cat(3, I, I, I);
%     imshow(rgbImage)
    imwrite(rgbImage, sprintf('data_rgb/benign/%d.png',k))
%     S(k).data = I; % optional, save data.
    
end

% Treating benign
D = 'data_png/malignant/';
S = dir(fullfile(D,'*.png')); % pattern to match filenames.
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    I = imread(F);
    rgbImage = cat(3, I, I, I);
%     imshow(rgbImage)
    imwrite(rgbImage, sprintf('data_rgb/malignant/%d.png',k))
%     S(k).data = I; % optional, save data.
    
end
