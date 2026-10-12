function extract_fig_tfm(fig_path, out_path)
%EXTRACT_FIG_TFM  Pull the TFM image out of a BRAIN-saved .fig into a plain .mat.
%
%   extract_fig_tfm(fig_path, out_path)
%
%   Writes cdata (dB image, rows = z), x_mm, z_mm, clim (display range) and cmap.
%   Needs the BRAIN toolbox on the path (the .fig carries BRAIN callbacks);
%   errors from its figure DeleteFcn on close are harmless.
f  = openfig(fig_path, 'invisible');
im = findobj(f, 'Type', 'image');
im = im(1);
ax = ancestor(im, 'axes');
out.cdata = double(im.CData);
out.x_mm  = im.XData(:)';
out.z_mm  = im.YData(:)';
out.clim  = ax.CLim;
out.cmap  = colormap(ax);
save(out_path, '-struct', 'out', '-v7');
close(f);
end
