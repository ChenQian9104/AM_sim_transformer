% for compressor 
filename = './STL_file/CompressorBlade.stl';
x = linspace(-100,100,224); y  = linspace(-100,100,224);z = 40/1000:10*40/1000:750*40/1000; 

result = VOXELISE(x,y,z,filename);
figure(1);imagesc( rot90(result(:,:,1)) ); axis equal; 
figure(2); imagesc(rot90(result(:,:,30)));axis equal; 
figure(3); imagesc(rot90(result(:,:,75)));axis equal; 

save('compressor.mat', 'result'); 
return

filename = './STL_file/GE_Prototype_1.stl';
x = linspace(-100,100,256); y  = linspace(-100,100,256);z = 40/1000:40/1000:1560*40/1000; 

result = VOXELISE(x,y,z,filename);
figure(1);imagesc( rot90(result(:,:,1)) ); axis equal; 
figure(2); imagesc(rot90(result(:,:,200)));axis equal; 
figure(3); imagesc(rot90(result(:,:,500)));axis equal; 
save('bracket.mat', 'result'); 
return
% for Blade
filename = './STL_file/Simple_Geometry_Contour_Blades.stl';
x = linspace(-100,100,256); y  = linspace(-100,100,256);z = 40/1000:40/1000:2560*40/1000; 
result = VOXELISE(x,y,z,filename);
save('blade.mat', 'result'); 

return; 
