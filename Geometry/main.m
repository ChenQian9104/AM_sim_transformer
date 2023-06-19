% for compressor 
filename = './STL_file/CompressorBlade.stl';
x = linspace(-100,100,224); y  = linspace(-100,100,224);z = 40/1000:25*40/1000:750*40/1000; 

result = VOXELISE(x,y,z,filename);
figure(1);imagesc( rot90(result(:,:,1)) ); axis equal; 
figure(2); imagesc(rot90(result(:,:,20)));axis equal; 
figure(3); imagesc(rot90(result(:,:,30)));axis equal; 

save('compressor.mat', 'result'); 


filename = './STL_file/GE_Prototype_1.stl';
x = linspace(-100,100,224); y  = linspace(-100,100,224);z = 40/1000:25*40/1000:1560*40/1000; 

result = VOXELISE(x,y,z,filename);
figure(4);imagesc( rot90(result(:,:,1)) ); axis equal; 
figure(5); imagesc(rot90(result(:,:,20)));axis equal; 
figure(6); imagesc(rot90(result(:,:,50)));axis equal; 
save('bracket.mat', 'result'); 

% for Blade
filename = './STL_file/Simple_Geometry_Contour_Blades_V2.stl';
x = linspace(-100,100,224); y  = linspace(-100,100,224);z = 40/1000:25*40/1000:2560*40/1000; 
 
result = VOXELISE(x,y,z,filename);
figure(7);imagesc( rot90(result(:,:,1)) ); axis equal; 
figure(8); imagesc(rot90(result(:,:,50)));axis equal; 
figure(9); imagesc(rot90(result(:,:,100)));axis equal;
save('blade.mat', 'result'); 




% for Column
filename = './STL_file/Column.stl';
x = linspace(-100,100,224); y  = linspace(-100,100,224);z = 40/1000:25*40/1000:750*40/1000; 
 
result = VOXELISE(x,y,z,filename);
figure(10);imagesc( rot90(result(:,:,1)) ); axis equal; 
figure(11); imagesc(rot90(result(:,:,20)));axis equal; 
figure(12); imagesc(rot90(result(:,:,30)));axis equal;
save('column.mat', 'result'); 

return; 

