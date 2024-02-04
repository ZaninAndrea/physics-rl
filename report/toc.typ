#let tocLine(header, inset: 0pt) = {
  let title = header.body.text
  let page = header.location().page() - 1

  let rawCounter = counter(heading).at(header.location())
  let headerCounter = h(14pt)
  if header.numbering != none{
    headerCounter = numbering(header.numbering, ..rawCounter)
  }

  let dots = box(width: 1fr, repeat[. #h(2pt)])
  [#h(inset) #headerCounter #title #dots #page #linebreak()]
}

#let toc = {
  // let tocHeading = heading("Table of Contents", numbering: none)
  // tocHeading
  set par(first-line-indent: 0pt)
  
  locate(loc => {
    // Find all top level headings
    let elems = query(
      selector(heading.where(level:1).after(loc)),
      loc
    )

    for (i, el) in elems.enumerate(){
      // if el == tocHeading{
      //   continue
      // }
      
      tocLine(el, inset: 0pt)

      // Find all sub-headings of this heading
      let children = selector(heading.where(level:2)).after(el.location())
      if i < elems.len() - 1 {
        children = children.before(elems.at(i+1).location())
      }
      let childElements = query(
        children,
        loc
      )

      // Display all children
      for child in childElements{
        tocLine(child, inset: 20pt)
      }
    } 
  })
}