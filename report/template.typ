#let script-size = 7.97224pt
#let footnote-size = 8.50012pt
#let small-size = 9.24994pt
#let normal-size = 12.0000pt
#let large-size = 15pt

// This function gets your whole document as its `body` and formats
// it as an article in the style of the American Mathematical Society.
#let ams-article(
  // The article's title.
  title: "Paper title",

  shortTitle: none,

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // Your article's abstract. Can be omitted if you don't have one.
  abstract: none,

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The path to a bibliography file if you want to cite some external
  // works.
  bibliography-file: none,

  // The document's content.
  body,
) = {
  // Formats the author's names in a list with commas and a
  // final "and".
  let names = authors.map(author => author.name)
  let author-string = if authors.len() == 2 {
    names.join(" and ")
  } else {
    names.join(", ", last: ", and ")
  }

  // Set document metadata.
  set document(title: title, author: names)

  // Set the body font. AMS uses the LaTeX font.
  set text(size: normal-size, font: "New Computer Modern")

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: if paper-size != "a4" {
      (
        top: (116pt / 279mm) * 100%,
        left: (126pt / 216mm) * 100%,
        right: (128pt / 216mm) * 100%,
        bottom: (94pt / 279mm) * 100%,
      )
    } else {
      (
        top: 100pt,
        left: 100pt,
        right: 100pt,
        bottom: 80pt,
      )
    },

    // The page header should show the page number and list of
    // authors, except on the first page. The page number is on
    // the left for even pages and on the right for odd pages.
    header-ascent: 30pt,
    header: locate(loc => {
      let i = counter(page).at(loc).first()
      if i == 1 { return }
      i -= 1

      let headerText = title
      if shortTitle != none{
        headerText = shortTitle
      }
      
      set text(size: small-size)
      grid(
        columns: (6em, 1fr, 6em),
        [#i],
        align(center, upper(headerText))
      )
    }),
  )

  // Configure headings.
  set heading(numbering: "1.")
  show heading: it => {
    // Create the heading numbering.
    let number = if it.numbering != none {
      counter(heading).display(it.numbering)
      h(7pt, weak: true)
    }

    // Level 1 headings are centered and smallcaps.
    // The other ones are run-in.
    set text(size: normal-size, weight: 400)
    if it.level == 1 {
      set align(center)
      set text(size: large-size)

      pagebreak(weak: true)
      strong([
        #number
        #it.body
      ])
      v(15pt, weak: true)
    } else {
      v(15pt, weak: true)
      number
      strong(it.body)
      h(7pt, weak: true)
    }
  }

  set text(hyphenate: false)

  // Configure lists and links.
  set list(indent: 24pt, body-indent: 5pt)
  set enum(indent: 24pt, body-indent: 5pt)
  show link: set text(font: "New Computer Modern Mono")

  // Configure equations.
  show math.equation: set block(below: 8pt, above: 9pt)
  show math.equation: set text(weight: 400)

  // Configure citation and bibliography styles.
  set bibliography(style: "springer-mathphys", title: "References")

  show figure: it => {
    set align(center)

    v(12.5pt, weak: true)

    // Display the figure's body.
    block(it.body, inset: (
      left: -60pt,
      right: -60pt
    ))

    // Display the figure's caption.
    if it.has("caption") {
      // Gap defaults to 17pt.
      v(if it.has("gap") { it.gap } else { 17pt }, weak: true)
      smallcaps(it.supplement)
      if it.numbering != none {
        [ ]
        it.counter.display(it.numbering)
      }
      [. ]
      it.caption.body
    }

    v(15pt, weak: false)
  }


  // Display the title and authors.
  v(35pt, weak: true)
  align(center, upper({
    text(size: 1.2 * normal-size, weight: 500, "PACS Project Report")
    v(25pt, weak: true)
    text(size: 1.3 * large-size, weight: 700, title)
    v(25pt, weak: true)
    text(size: normal-size, author-string)
  }))

  // Configure paragraph properties.
  set par(first-line-indent: 0em, justify: true, leading: 0.58em)
  show par: set block(spacing: 0.58em, above: 1em)

  // Display the abstract
  if abstract != none {
    v(20pt, weak: true)
    set text(script-size)
    show: pad.with(x: 35pt)
    smallcaps[Abstract. ]
    abstract
  }

  // Display the article's contents.
  v(60pt, weak: true)
  body

  // Display the bibliography, if any is given.
  if bibliography-file != none {
    show bibliography: set text(8.5pt)
    show bibliography: pad.with(x: 0.5pt)
    bibliography(bibliography-file)
  }
}