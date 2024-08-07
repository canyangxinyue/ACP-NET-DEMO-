

import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import matplotlib
import cv2
import numpy as np

def add_right_cax(fig, ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axposu = ax[0,-1].get_position()
    axposd = ax[-1,-1].get_position()
    caxpos = matplotlib.transforms.Bbox.from_extents(
        axposu.x1 + pad,
        axposd.y0,
        axposu.x1 + pad + width,
        axposu.y1
    )
    cax = ax[0,0].figure.add_axes(caxpos)

    return cax
    
def save_train_image(name,data,output_dir='fig'):
    pic,axs=plt.subplots(2,3)
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.25)
    # img threshold_map threshold_mask shrink_map shrink_mask
    axs[0,0].set_title(data['img_name'])
    origin_image=(data['img'][0]*255).numpy().astype('uint8').transpose(1,2,0)
    axs[0,0].imshow(origin_image)
    axs[1,0].set_title('labels')
    # for poly in data['text_polys']:
    #     image=cv2.polylines(origin_image, [np.array(poly,dtype=np.int32).reshape(-1,1,2)], isClosed=True, color=(255,0,0), thickness=3)
    axs[1,0].imshow((origin_image*data['threshold_mask'].numpy().transpose(1,2,0)).astype('uint8'))
    if 'text_polys' in data.keys():#画框
        for poly in data['text_polys']:  
            poly = [[int(p[0][0]),int(p[1][0])] for p in poly]
            axs[1,0].add_patch(patches.Polygon(xy=poly, color='red', alpha=0.8, fill=False))
        
    axs[0,1].set_title('threshold_map')
    axs[0,1].imshow(data['threshold_map'][0],vmin=0.,vmax=1.)
    axs[0,1].axis('off')
    axs[0,2].set_title('threshold_mask')
    axs[0,2].imshow(data['threshold_mask'][0],vmin=0.,vmax=1.)
    axs[0,2].axis('off')
    axs[1,1].set_title('shrink_map')
    axs[1,1].imshow(data['shrink_map'][0],vmin=0.,vmax=1.)
    axs[1,1].axis('off')
    axs[1,2].set_title('shrink_mask')
    axs[1,2].imshow(data['shrink_mask'][0],vmin=0.,vmax=1.)
    axs[1,2].axis('off')
    pic.colorbar(plt.cm.ScalarMappable(),cax=add_right_cax(pic,axs,0.02,0.02),extend='both',shrink=0.7)
    plt.savefig(f"{output_dir}/{name}.jpg",bbox_inches = 'tight',dpi=800)
    plt.close()

def save_train_distance_image(name,data,output_dir='fig'):
    pic,axs=plt.subplots(2,4)
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.25)
    # img threshold_map threshold_mask shrink_map shrink_mask
    axs[0,0].set_title(data['img_name'])
    origin_image=(data['img'][0]*255).numpy().astype('uint8').transpose(1,2,0)
    axs[0,0].imshow(origin_image)
    axs[1,0].set_title('labels')
    # for poly in data['text_polys']:
    #     image=cv2.polylines(origin_image, [np.array(poly,dtype=np.int32).reshape(-1,1,2)], isClosed=True, color=(255,0,0), thickness=3)
    axs[1,0].imshow((origin_image*data['threshold_mask'].numpy().transpose(1,2,0)).astype('uint8'))
    if 'text_polys' in data.keys():#画框
        for poly in data['text_polys']:  
            poly = [[int(p[0][0]),int(p[1][0])] for p in poly]
            axs[1,0].add_patch(patches.Polygon(xy=poly, color='red', alpha=0.8, fill=False))
        
    axs[0,1].set_title('threshold_map')
    axs[0,1].imshow(data['threshold_map'][0],vmin=0.,vmax=1.)
    axs[0,1].axis('off')
    axs[0,2].set_title('threshold_mask')
    axs[0,2].imshow(data['threshold_mask'][0],vmin=0.,vmax=1.)
    axs[0,2].axis('off')
    axs[1,1].set_title('shrink_map')
    axs[1,1].imshow(data['shrink_map'][0],vmin=0.,vmax=1.)
    axs[1,1].axis('off')
    axs[1,2].set_title('shrink_mask')
    axs[1,2].imshow(data['shrink_mask'][0],vmin=0.,vmax=1.)
    axs[1,2].axis('off')
    axs[0,3].set_title('gt_distances')
    axs[0,3].imshow(data['gt_distances'][0,0],vmin=0.,vmax=1.)
    axs[0,3].axis('off')
    axs[1,3].set_title('gt_kernel_instances')
    h,w=data['gt_distances'].shape[-2:]
    x = np.arange(0, w, 4)
    y = np.arange(0, h, 4)
    axs[1,3].invert_yaxis()
    axs[1,3].quiver(x, y, data['gt_distances'][0,0,::4,::4], -data['gt_distances'][0,1,::4,::4],scale=0.2,units='xy')
    axs[1,3].set_aspect('equal')
    # axs[1,3].axis('off')
    pic.colorbar(plt.cm.ScalarMappable(),cax=add_right_cax(pic,axs,0.02,0.02),extend='both',shrink=0.7)
    plt.savefig(f"{output_dir}/{name}.jpg",bbox_inches = 'tight',dpi=1600)
    plt.close()

def save_distance_image(name,data,output_dir='fig'):
    pic,axs=plt.subplots(1)
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.05,hspace=0.25)
  
    h,w=data['gt_distances'].shape[-2:]
    distance_map=data['gt_distances'][0,:,::4,::4]
    x = np.arange(0, w, 4)
    y = np.arange(0, h, 4)
    mask = (distance_map[0]!=0)|(distance_map[1]!=0)
    axs.invert_yaxis()
    axs.quiver(x[mask], y[mask], distance_map[0][mask], -distance_map[1][mask],scale=0.2,units='xy',linewidths=1)
    axs.set_aspect('equal')
    plt.savefig(f"{output_dir}/{name}.jpg",bbox_inches = 'tight',dpi=800)
    plt.close()

def save_test_image(name,data,output_dir='fig'):
    image=(data['img'][0]*255).numpy().round().astype('uint8').transpose(1,2,0)
    for poly in data['text_polys']:
        image=cv2.polylines(image, [np.array(poly,dtype=np.int32).reshape(-1,1,2)], isClosed=True, color=(255,0,0), thickness=3)
    # cv2.imwrite(f"fig/{name}.jpg",image)
    plt.title(data['img_name'])
    plt.imshow(image.get())
    plt.savefig(f"{output_dir}/{name}.jpg",bbox_inches = 'tight')
    plt.close()
    
def save_rec_image(name,data,output_dir='fig'):
    image=(data['img'][0]*255).numpy().round().astype('uint8').transpose(1,2,0)
    # cv2.imwrite(f"fig/{name}.jpg",image)
    plt.title(data['img_name'][0]+"-"+data['texts'][0][0])
    plt.imshow(image)
    plt.savefig(f"{output_dir}/{name}.jpg")
    plt.close()