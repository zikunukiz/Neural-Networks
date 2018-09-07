import os
import webbrowser

def gen_html_template(img_list, caption_list):

    html_text = """
    <html>
    <head>
    <style>

    figure {
      float: left;
      width: 50%;
      text-align: center;
      font-style: italic;
      font-size: smaller;
      text-indent: 0;
      border: thin silver solid;
      margin: 0.5em;
      padding: 0.5em;
      display: table;
    }

    figcaption {
      display: table-caption;
      caption-side: top;
    }

    </style>
    </head>
    <body>
    """

    for ii, cc in zip(img_list, caption_list):
        html_text += '<figure>\n'
        html_text += '\t<img class=scaled src="{}" alt="{}">\n'.format(ii, ii)
        html_text += '\t<figcaption>{}</figcaption>\n'.format(cc)
        html_text += '</figure>\n'
        
    html_text += """
    </body>
    </html>
    """

    return html_text

def show_in_browser(img_list, caption_list):

    save_name = 'html/image_caption.html'
    f = open(save_name,'w')

    html_text = gen_html_template(img_list, caption_list)

    f.write(html_text)
    f.close()

    #Change path to reflect file location
    html_file = os.path.join('file:///' + os.getcwd(), save_name)
    webbrowser.open_new_tab(html_file)


if __name__ == '__main__':

    img_list = ['COCO_val2014_000000145781.jpg', 'COCO_val2014_000000145781.jpg']
    caption_list = ['a', 'b']

    show_in_browser(img_list, caption_list)
